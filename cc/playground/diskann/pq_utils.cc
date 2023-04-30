// Borrowed from github.com/microsoft/diskann
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
// Modified by: Suhas Jayaram Subramanya (suhasj@cs.cmu.edu)

#include <cstdint>
#include <immintrin.h>
#include <unordered_set>
#include <vector>

#include "pq_utils.h"

// block size for reading/processing large files and matrices in blocks
#define BLOCK_SIZE 5000000

namespace {
uint64_t get_file_size(const std::string &path) {
  std::ifstream in(path, std::ifstream::ate | std::ifstream::binary);
  return (uint64_t)in.tellg();
}
} // namespace

namespace diskann {

void FixedChunkPQTable::load_pq_centroid_bin(const std::string &pq_table_file,
                                             uint64_t num_chunks) {

  // compute memory requirement & allocate (nearest page size)
  uint64_t alloc_size = ::get_file_size(pq_table_file);
  // 2x to account for transpose matrix
  alloc_size = 2 * ROUND_UP(alloc_size, 256);
  std::cout << "Allocating " << alloc_size << "B for PQ table" << std::endl;
  this->buf = (uint8_t *)FASTER::core::aligned_alloc(256, alloc_size);

  // read & verify metadata
  uint8_t *cur_buf = this->buf;
  uint32_t nr, nc;
  diskann::get_bin_metadata(pq_table_file, nr, nc);
  std::vector<uint64_t> offset_data(nr * nc + 16);
  diskann::populate_from_bin<uint64_t>(offset_data.data(), pq_table_file, nr,
                                       nc);
  if (nr != 4) {
    std::cout << "Error reading pq_pivots file " << pq_table_file
              << ". Offsets dont contain correct metadata, # offsets = " << nr
              << ", but expected " << 4;
    exit(-1);
  }

  std::cout << "Offsets: " << offset_data[0] << " " << offset_data[1] << " "
            << offset_data[2] << " " << offset_data[3] << std::endl;

  // populate tables
  diskann::get_bin_metadata(pq_table_file, nr, nc, offset_data[0]);
  if ((nr != NUM_PQ_CENTROIDS)) {
    std::cout << "Error reading pq_pivots file " << pq_table_file
              << ". file_num_centers  = " << nr << " but expecting "
              << NUM_PQ_CENTROIDS << " centers";
  }
  std::cout << "Loading PQ Pivots: #ctrs: " << NUM_PQ_CENTROIDS
            << ", #dims: " << nc << std::endl;
  // allocate tables, align to 256 bytes
  cur_buf = (uint8_t *)ROUND_UP((uint64_t)cur_buf, 256);
  this->tables = (float *)cur_buf;
  diskann::populate_from_bin<float>(this->tables, pq_table_file, nr, nc, nc,
                                    offset_data[0]);
  cur_buf += (nr * nc * sizeof(float));
  this->ndims = nc;

  // populate centroid
  diskann::get_bin_metadata(pq_table_file, nr, nc, offset_data[1]);
  if ((nr != this->ndims) || (nc != 1)) {
    std::cerr << "Error reading centroids from pq_pivots file " << pq_table_file
              << ". file_dim  = " << nr << ", file_cols = " << nc
              << " but expecting " << this->ndims << " entries in 1 dimension.";
  }
  std::cout << "Loading PQ centroid: #centroids: " << 1
            << ", #dims: " << this->ndims << std::endl;
  // allocate centroid, align to 256 bytes
  cur_buf = (uint8_t *)ROUND_UP((uint64_t)cur_buf, 256);
  this->centroid = (float *)cur_buf;
  diskann::populate_from_bin<float>(this->centroid, pq_table_file, nr, nc, nc,
                                    offset_data[1]);
  cur_buf += (nr * nc * sizeof(float));

  int chunk_offsets_index = 2;
  // populate chunk offsets
  diskann::get_bin_metadata(pq_table_file, nr, nc,
                            offset_data[chunk_offsets_index]);
  if (nc != 1 || (nr != num_chunks + 1 && num_chunks != 0)) {
    std::cerr << "Error loading chunk offsets file. numc: " << nc
              << " (should be 1). numr: " << nr << " (should be "
              << num_chunks + 1 << " or 0 if we need to infer)" << std::endl;
  }
  std::cout << "Loading PQ chunk offsets: #offsets: " << nr << ", #dims: " << nc
            << std::endl;
  // allocate chunk offsets, align to 256 bytes
  cur_buf = (uint8_t *)ROUND_UP((uint64_t)cur_buf, 256);
  this->chunk_offsets = (uint32_t *)cur_buf;
  diskann::populate_from_bin<uint32_t>(this->chunk_offsets, pq_table_file, nr,
                                       nc, nc,
                                       offset_data[chunk_offsets_index]);
  cur_buf += (nr * nc * sizeof(uint32_t));
  this->n_chunks = nr - 1;
  std::cout << "Loaded PQ Pivots: #ctrs: " << NUM_PQ_CENTROIDS
            << ", #dims: " << this->ndims << ", #chunks: " << this->n_chunks
            << std::endl;

  // alloc and compute transpose, align to 256 bytes
  cur_buf = (uint8_t *)ROUND_UP((uint64_t)cur_buf, 256);
  this->tables_tr = (float *)cur_buf;
  for (uint64_t i = 0; i < 256; i++) {
    for (uint64_t j = 0; j < this->ndims; j++) {
      this->tables_tr[j * 256 + i] = this->tables[i * this->ndims + j];
    }
  }
  cur_buf += (256 * this->ndims * sizeof(float));

  // safe write to last element: should not trigger seg-fault
  *(uint64_t *)cur_buf = 0ul;
}

uint32_t FixedChunkPQTable::get_num_chunks() { return n_chunks; }

FixedChunkPQTable::~FixedChunkPQTable() {
  if (this->buf != nullptr) {
    FASTER::core::aligned_free(this->buf);
  }
}
void FixedChunkPQTable::preprocess_query(float *query_vec) {
  for (uint32_t d = 0; d < ndims; d++) {
    query_vec[d] -= this->centroid[d];
  }
}

// assumes pre-processed query
void FixedChunkPQTable::populate_chunk_distances(const float *query_vec,
                                                 float *dist_vec) {
  memset(dist_vec, 0, 256 * this->n_chunks * sizeof(float));
  // chunk wise distance computation
  for (uint64_t chunk = 0; chunk < this->n_chunks; chunk++) {
    // sum (q-c)^2 for the dimensions associated with this chunk
    float *chunk_dists = dist_vec + (256 * chunk);
    for (uint64_t j = this->chunk_offsets[chunk];
         j < this->chunk_offsets[chunk + 1]; j++) {
      const float *centers_dim_vec = this->tables_tr + (256 * j);
      for (uint64_t idx = 0; idx < 256; idx++) {
        double diff = centers_dim_vec[idx] - (query_vec[j]);
        chunk_dists[idx] += (float)(diff * diff);
      }
    }
  }
}

float FixedChunkPQTable::l2_distance(const float *query_vec,
                                     uint8_t *base_vec) {
  float res = 0;
  for (uint64_t chunk = 0; chunk < this->n_chunks; chunk++) {
    for (uint64_t j = this->chunk_offsets[chunk];
         j < this->chunk_offsets[chunk + 1]; j++) {
      const float *centers_dim_vec = this->tables_tr + (256 * j);
      float diff = centers_dim_vec[base_vec[chunk]] - (query_vec[j]);
      res += diff * diff;
    }
  }
  return res;
}

float FixedChunkPQTable::inner_product(const float *query_vec,
                                       uint8_t *base_vec) {
  float res = 0;
  for (uint64_t chunk = 0; chunk < this->n_chunks; chunk++) {
    for (uint64_t j = this->chunk_offsets[chunk];
         j < this->chunk_offsets[chunk + 1]; j++) {
      const float *centers_dim_vec = this->tables_tr + (256 * j);
      float diff = centers_dim_vec[base_vec[chunk]] *
                   query_vec[j]; // assumes centroid is 0 to
                                 // prevent translation errors
      res += diff;
    }
  }
  return -res; // returns negative value to simulate distances (max -> min
               // conversion)
}

// assumes no rotation is involved
void FixedChunkPQTable::inflate_vector(uint8_t *base_vec, float *out_vec) {
  for (uint64_t chunk = 0; chunk < this->n_chunks; chunk++) {
    for (uint64_t j = this->chunk_offsets[chunk];
         j < this->chunk_offsets[chunk + 1]; j++) {
      const float *centers_dim_vec = this->tables_tr + (256 * j);
      out_vec[j] = centers_dim_vec[base_vec[chunk]] + centroid[j];
    }
  }
}

void FixedChunkPQTable::populate_chunk_inner_products(const float *query_vec,
                                                      float *dist_vec) {
  memset(dist_vec, 0, 256 * this->n_chunks * sizeof(float));
  // chunk wise distance computation
  for (uint64_t chunk = 0; chunk < this->n_chunks; chunk++) {
    // sum (q-c)^2 for the dimensions associated with this chunk
    float *chunk_dists = dist_vec + (256 * chunk);
    for (uint64_t j = this->chunk_offsets[chunk];
         j < this->chunk_offsets[chunk + 1]; j++) {
      const float *centers_dim_vec = this->tables_tr + (256 * j);
      for (uint64_t idx = 0; idx < 256; idx++) {
        double prod =
            centers_dim_vec[idx] * query_vec[j]; // assumes that we are not
                                                 // shifting the vectors to
                                                 // mean zero, i.e., centroid
                                                 // array should be all zeros
        chunk_dists[idx] -=
            (float)prod; // returning negative to keep the search code
                         // clean (max inner product vs min distance)
      }
    }
  }
}
void aggregate_coords(const uint32_t *ids, const uint64_t n_ids,
                      const uint8_t *all_coords, const uint64_t nchunks,
                      uint8_t *out, const uint64_t aligned_nchunks) {
  uint64_t chunks_per_row = (aligned_nchunks == 0) ? nchunks : aligned_nchunks;
  for (uint64_t i = 0; i < n_ids; i++) {
    const uint8_t *in_ptr = all_coords + ids[i] * chunks_per_row;
    uint8_t *out_ptr = out + i * chunks_per_row;
    std::copy(in_ptr, in_ptr + nchunks, out_ptr);
  }
}

void pq_dist_lookup(const uint8_t *pq_ids, const uint64_t n_pts,
                    const uint64_t nchunks, const float *pq_dists,
                    float *dists_out, const uint64_t aligned_nchunks) {
  _mm_prefetch((char *)dists_out, _MM_HINT_T0);
  _mm_prefetch((char *)pq_ids, _MM_HINT_T0);
  _mm_prefetch((char *)(pq_ids + 64), _MM_HINT_T0);
  _mm_prefetch((char *)(pq_ids + 128), _MM_HINT_T0);
  std::fill(dists_out, dists_out + n_pts, 0);
  uint64_t chunks_per_row = (aligned_nchunks == 0) ? nchunks : aligned_nchunks;
  for (uint64_t chunk = 0; chunk < nchunks; chunk++) {
    const float *chunk_dists = pq_dists + 256 * chunk;
    if (chunk < nchunks - 1) {
      _mm_prefetch((char *)(chunk_dists + 256), _MM_HINT_T0);
    }
    for (uint64_t idx = 0; idx < n_pts; idx++) {
      uint8_t pq_centerid = pq_ids[chunks_per_row * idx + chunk];
      dists_out[idx] += chunk_dists[pq_centerid];
    }
  }
}
} // namespace diskann
