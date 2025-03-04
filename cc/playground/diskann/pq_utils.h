// Borrowed from github.com/microsoft/diskann
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
// Modified by: Suhas Jayaram Subramanya (suhasj@cs.cmu.edu)
#pragma once

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>

#include "../src/core/faster.h"
#include "io_utils.h"

#define NUM_PQ_BITS 8
#define NUM_PQ_CENTROIDS (1 << NUM_PQ_BITS)
#define MAX_OPQ_ITERS 20
#define NUM_KMEANS_REPS_PQ 12
#define MAX_PQ_TRAINING_SET_SIZE 256000
#define MAX_PQ_CHUNKS 64

namespace diskann {
class FixedChunkPQTable {
  float *tables = nullptr; // pq_tables = float array of size [256 * ndims]
  uint64_t ndims = 0;      // ndims = true dimension of vectors
  uint64_t n_chunks = 0;
  bool use_rotation = false;
  uint32_t *chunk_offsets = nullptr;
  float *centroid = nullptr;
  float *tables_tr = nullptr; // same as pq_tables, but col-major
  float *rotmat_tr = nullptr;

  // backing buf for all the above pointers
  uint8_t *buf = nullptr;

public:
  FixedChunkPQTable() {}
  ~FixedChunkPQTable();

  void load_pq_centroid_bin(const std::string &pq_table_file,
                            size_t num_chunks);

  uint32_t get_num_chunks();

  void preprocess_query(float *query_vec);

  // assumes pre-processed query
  void populate_chunk_distances(const float *query_vec, float *dist_vec);

  float l2_distance(const float *query_vec, uint8_t *base_vec);

  float inner_product(const float *query_vec, uint8_t *base_vec);

  // assumes no rotation is involved
  void inflate_vector(uint8_t *base_vec, float *out_vec);

  void populate_chunk_inner_products(const float *query_vec, float *dist_vec);
};

template <typename T> struct PQScratch {
  float *aligned_pqtable_dist_scratch =
      nullptr;                           // MUST BE AT LEAST [256 * NCHUNKS]
  float *aligned_dist_scratch = nullptr; // MUST BE AT LEAST MAX_DEGREE
  uint8_t *aligned_pq_coord_scratch =
      nullptr; // MUST BE AT LEAST  [N_CHUNKS * MAX_DEGREE]
  float *rotated_query = nullptr;
  float *aligned_query_float = nullptr;
  void *aligned_buf = nullptr; // backing buf for all pointers

  static uint64_t get_alloc_size(uint32_t max_degree, uint32_t aligned_dim) {
    // aligned_pq_coord_scratch
    uint64_t alloc_size =
        (uint64_t)max_degree * (uint64_t)MAX_PQ_CHUNKS * sizeof(uint8_t);
    // aligned_pqtable_dist_scratch
    alloc_size += 256 * (uint64_t)MAX_PQ_CHUNKS * sizeof(float);
    // aligned_dist_scratch
    alloc_size += ROUND_UP((uint64_t)max_degree * sizeof(float), 256);
    // aligned_query_float
    alloc_size += ROUND_UP(aligned_dim * sizeof(float), 256);
    // rotated_query
    alloc_size += ROUND_UP(aligned_dim * sizeof(float), 256);
    return alloc_size;
  }

  PQScratch(uint32_t max_degree, uint32_t aligned_dim) {
    uint64_t alloc_size = PQScratch<T>::get_alloc_size(max_degree, aligned_dim);
    this->aligned_buf = FASTER::core::aligned_alloc(256, alloc_size);
    memset(this->aligned_buf, 0, alloc_size);
    uint8_t *cur_buf = (uint8_t *)this->aligned_buf;

    // pq coord scratch
    this->aligned_pq_coord_scratch = cur_buf;
    cur_buf += (uint64_t)max_degree * (uint64_t)MAX_PQ_CHUNKS * sizeof(uint8_t);

    // pqtable dist scratch
    this->aligned_pqtable_dist_scratch = (float *)cur_buf;
    cur_buf += 256 * (uint64_t)MAX_PQ_CHUNKS * sizeof(float);

    // dist scratch
    this->aligned_dist_scratch = (float *)cur_buf;
    cur_buf += ROUND_UP((uint64_t)max_degree * sizeof(float), 256);

    // query float
    this->aligned_query_float = (float *)cur_buf;
    cur_buf += ROUND_UP(aligned_dim * sizeof(float), 256);

    // rotated query
    this->rotated_query = (float *)cur_buf;
    cur_buf += ROUND_UP(aligned_dim * sizeof(float), 256);

    // sanity check
    assert(cur_buf == (uint8_t *)this->aligned_buf + alloc_size);
  }

  ~PQScratch() { FASTER::core::aligned_free(this->aligned_buf); }

  void set(size_t dim, const T *query, const float norm = 1.0f) {
    for (size_t d = 0; d < dim; ++d) {
      if (norm != 1.0f)
        rotated_query[d] = aligned_query_float[d] =
            static_cast<float>(query[d]) / norm;
      else
        rotated_query[d] = aligned_query_float[d] =
            static_cast<float>(query[d]);
    }
  }
};

void aggregate_coords(const uint32_t *ids, const uint64_t n_ids,
                      const uint8_t *all_coords, const uint64_t nchunks,
                      uint8_t *out, const uint64_t aligned_nchunks = 0);

void pq_dist_lookup(const uint8_t *pq_ids, const size_t n_pts,
                    const size_t nchunks, const float *pq_dists,
                    float *dists_out, const uint64_t aligned_nchunks = 0);

} // namespace diskann
