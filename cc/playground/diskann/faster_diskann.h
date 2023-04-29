// Implementation of a zero-additional memory DiskANN index (purely on-disk,
// using a FASTER FilesystemDisk)
// Author: Suhas Jayaram Subramanya (suhasj@cs.cmu.edu)

#pragma once
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <immintrin.h>
#include <iostream>
#include <string>
#include <unordered_set>

#include "disk_graph.h"
#include "io_utils.h"
#include "pq_utils.h"
#include "search_utils.h"

#include "../src/core/faster.h"

#define MIN_FASTER_LOG_SIZE (1 << 30)  // 1 GB
#define FASTER_LOG_ALIGNMENT (1 << 25) // 32 MB
#define MAX_BEAM_WIDTH (1 << 16)       // max beam width for beam search
#define MAX_VAMANA_DEGREE (1 << 7)     // max degree for Vamana graph

namespace diskann {
class FasterDiskANNIndex {
private:
  // basic index params
  uint64_t num_points_ = 0, dim_ = 0, aligned_dim_ = 0;
  uint64_t max_degree_ = 0;
  uint64_t diskann_nodesize_ = 0;
  diskann::DiskGraph *graph_ = nullptr;
  uint32_t start_ = 0;

  // PQ params
  uint8_t *pq_data_ = nullptr;
  uint64_t pq_dim_ = 0, pq_aligned_dim_ = 0;
  FixedChunkPQTable pq_table;

  // index parameters
  std::string index_load_path_ = "";

  // FASTER store params
  uint64_t faster_max_keys_ = 0;
  uint64_t faster_memory_size_ = 0;
  std::string faster_graph_path_ = "/dev/shm/diskann_zero.store";

  // extra config params
  // whether to verify after loading graph data
  bool verify_after_load_ = true;

  // cache data for some nodes
  // map from { node id -> (flat_arr_idx, num_nbrs) }
  std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> cached_node_ids_;
  float *cached_node_data_buf = nullptr;
  uint64_t cached_node_data_size = 0; // in bytes
  uint64_t cached_node_nbrs_size = 0; // in bytes
  uint32_t *cached_node_nbrs_buf = nullptr;

  uint8_t *get_pq_vec(uint32_t node_id) {
    return this->pq_data_ + (uint64_t)node_id * this->pq_aligned_dim_;
  }

public:
  FasterDiskANNIndex(const std::string &index_load_path)
      : index_load_path_{index_load_path} {
    /*** 1. Load metadata for index ****/
    std::string disk_index_path = index_load_path + "_disk.index";
    uint32_t md_size, md_dim;
    diskann::get_bin_metadata(disk_index_path, md_size, md_dim);
    std::vector<uint64_t> md_vals(md_size + 8, 0);
    diskann::populate_from_bin<uint64_t>(md_vals.data(), disk_index_path,
                                         md_size, 1, 1);
    // populate metadata
    this->num_points_ = md_vals[0];
    this->dim_ = md_vals[1];
    // round up dim to multiple of 16 (good alignment for AVX ops)
    this->aligned_dim_ = ROUND_UP(this->dim_, 16);
    this->start_ = md_vals[2];
    this->diskann_nodesize_ = md_vals[3];
    this->max_degree_ =
        (this->diskann_nodesize_ - (this->dim_ * sizeof(float))) /
            sizeof(uint32_t) -
        1;
    std::cout << "Loaded metadata for index: " << std::endl;
    std::cout << "Data: num_points = " << this->num_points_
              << ", dim = " << this->dim_
              << ", aligned_dim = " << this->aligned_dim_ << std::endl;
    std::cout << "Index: start = " << this->start_
              << ", nodesize = " << this->diskann_nodesize_
              << ", max_degree = " << this->max_degree_ << std::endl;

    /*** 2. Configure FASTER store ****/
    // set max number of keys to be nearest power of 2 >= this->num_points_
    this->faster_max_keys_ = 1;
    while (this->faster_max_keys_ < this->num_points_)
      this->faster_max_keys_ <<= 1;
    std::cout << "Setting FASTER store max keys to " << this->faster_max_keys_
              << " (nearest power of 2 >= " << this->num_points_ << ")"
              << std::endl;
    // compute memory requirement for FASTER store
    uint64_t per_key_memory =
        sizeof(diskann::DiskannValue<float>) + // size of value obj
        this->diskann_nodesize_;               // data from disk
    per_key_memory +=
        sizeof(diskann::FixedSizeKey<uint32_t>); // size of key obj
    // round to nearest multiple of 8
    per_key_memory = ROUND_UP(per_key_memory, 16);
    this->faster_memory_size_ = this->faster_max_keys_ * per_key_memory;
    // round up to nearest 32 MB region
    this->faster_memory_size_ =
        ROUND_UP(this->faster_memory_size_, FASTER_LOG_ALIGNMENT);
    // min FASTER log size is 1 GB
    if (this->faster_memory_size_ < MIN_FASTER_LOG_SIZE)
      this->faster_memory_size_ = MIN_FASTER_LOG_SIZE;
    std::cout << "Configuring FASTER store memory size to "
              << this->faster_memory_size_ / (1 << 20)
              << " MB, per key memory = " << per_key_memory << "B" << std::endl;

    /*** 3. Create FASTER store ****/
    this->graph_ = new diskann::DiskGraph(this->faster_max_keys_,
                                          this->faster_memory_size_,
                                          this->faster_graph_path_);
    std::cout << "Created FASTER store for VamanaIndex" << std::endl;
    // std::cout << "Finished configuring index. Call load() to load graph into
    // FASTER store. " << std::endl;
  }

  uint64_t get_max_degree() { return this->max_degree_; }
  uint64_t get_num_points() { return this->num_points_; }
  uint64_t get_dim() { return this->dim_; }
  uint64_t get_aligned_dim() { return this->aligned_dim_; }

  // separated load() to call StartSession() before upserting graph data
  void load() {
    /*** 4. Load graph data ****/
    std::cout << "Loading graph data into FASTER store" << std::endl;
    std::string disk_index_path = this->index_load_path_ + "_disk.index";
    std::ifstream graph_reader(disk_index_path,
                               std::ios::in | std::ios::binary);
    // seek one sector to skip header
    graph_reader.seekg(4096, std::ios::beg);
    // read and upsert graph data
    uint32_t node_id, num_nbrs;
    uint8_t *aligned_sector_buf =
        (uint8_t *)FASTER::core::aligned_alloc(4096, 4096);
    uint32_t *node_nbrs = nullptr;
    float *node_data = nullptr;
    uint64_t sector_buf_offset = 0;
    // read first sector into buffer
    graph_reader.read(reinterpret_cast<char *>(aligned_sector_buf), 4096);

    for (uint32_t i = 0; i < this->num_points_; i++) {
      node_id = i;
      uint8_t *node_buf = aligned_sector_buf + sector_buf_offset;
      // vector data
      node_data = reinterpret_cast<float *>(node_buf);
      node_buf += this->dim_ * sizeof(float);
      // num neighbors
      num_nbrs = *(reinterpret_cast<uint32_t *>(node_buf));
      node_buf += sizeof(uint32_t);
      // read neighbros
      node_nbrs = reinterpret_cast<uint32_t *>(node_buf);
      // upsert into FASTER store
      // std::cout << "Upserting node " << node_id << " with degree: " <<
      // num_nbrs << std::endl;
      this->Upsert(node_id, node_data, this->dim_, node_nbrs, num_nbrs);

      // update max degree
      this->max_degree_ = std::max(this->max_degree_, (uint64_t)num_nbrs);

      // update sector offset
      sector_buf_offset += this->diskann_nodesize_;

      // check if next node is in cur sector buf
      if (sector_buf_offset + this->diskann_nodesize_ > 4096) {
        // read next sector into buffer
        graph_reader.read(reinterpret_cast<char *>(aligned_sector_buf), 4096);
        sector_buf_offset = 0;
      }
    }

    // round up max degree to nearest multiple of 8
    std::cout << "Max degree = " << this->max_degree_ << std::endl;
    std::cout << "Finished upserting index into FASTER store" << std::endl;

    // go back and verify all inserted data
    if (this->verify_after_load_) {
      std::cout << "Verifying inserted data" << std::endl;
      uint32_t num_nbrs_f;
      std::vector<uint32_t> node_nbrs_f(2 * this->max_degree_, 0);
      std::vector<float> node_data_f(2 * this->aligned_dim_, 0.0f);

      // reset file pointer to second sector; read sector into buf
      graph_reader.close();
      graph_reader.open(disk_index_path, std::ios::in | std::ios::binary);
      graph_reader.seekg(4096, std::ios::beg);
      graph_reader.read(reinterpret_cast<char *>(aligned_sector_buf), 4096);
      sector_buf_offset = 0;

      // read graph data
      for (uint32_t i = 0; i < this->num_points_; i++) {
        node_id = i;
        uint8_t *node_buf = aligned_sector_buf + sector_buf_offset;
        // vector data from disk
        node_data = reinterpret_cast<float *>(node_buf);
        node_buf += this->dim_ * sizeof(float);
        // num neighbors from disk
        num_nbrs = *(reinterpret_cast<uint32_t *>(node_buf));
        node_buf += sizeof(uint32_t);
        // neighbors from disk
        node_nbrs = reinterpret_cast<uint32_t *>(node_buf);
        // read both from FASTER store
        uint32_t read_dim = 0;
        this->Read(node_id, node_data_f.data(), read_dim, node_nbrs_f.data(),
                   num_nbrs_f);
        // verify metadata for node
        if (num_nbrs != num_nbrs_f) {
          std::cout << "ERROR: num_nbrs mismatch for node " << node_id
                    << " (disk = " << num_nbrs << ", FASTER = " << num_nbrs_f
                    << ")" << std::endl;
          exit(1);
        }
        if (read_dim != this->dim_) {
          std::cout << "ERROR: read_dim mismatch for node " << node_id
                    << " (disk = " << this->dim_ << ", FASTER = " << read_dim
                    << ")" << std::endl;
          exit(1);
        }
        // verify neighbors
        for (uint32_t j = 0; j < num_nbrs; j++) {
          if (node_nbrs[j] != node_nbrs_f[j]) {
            std::cout << "ERROR: nbrs mismatch for node " << node_id
                      << " (disk = " << node_nbrs[j]
                      << ", FASTER = " << node_nbrs_f[j] << ")" << std::endl;
            exit(1);
          }
        }
        // verify data
        if (memcmp(node_data, node_data_f.data(), this->dim_ * sizeof(float)) !=
            0) {
          std::cout << "ERROR: data mismatch for node " << node_id << std::endl;
          // print out mem data and disk data for debugging
          std::cout << "Mem vector = ";
          for (uint32_t j = 0; j < this->dim_; j++) {
            std::cout << node_data[j] << ",";
          }
          std::cout << std::endl;
          std::cout << "Disk vector = ";
          for (uint32_t j = 0; j < this->dim_; j++) {
            std::cout << node_data_f[j] << ",";
          }
          std::cout << std::endl;
          exit(1);
        }

        // update sector offset
        sector_buf_offset += this->diskann_nodesize_;

        // check if next node is in cur sector buf
        if (sector_buf_offset + this->diskann_nodesize_ > 4096) {
          // read next sector into buffer
          graph_reader.read(reinterpret_cast<char *>(aligned_sector_buf), 4096);
          sector_buf_offset = 0;
        }
      }
      // verification successful
      std::cout << "Verification successful" << std::endl;
    }

    std::cout << "Finished loading graph data and verifying insertion into "
                 "FASTER store"
              << std::endl;

    // close graph file
    graph_reader.close();

    // free aligned sector buffer
    FASTER::core::aligned_free(aligned_sector_buf);
  }

  void load_pq() {
    std::string pq_pivots_path = this->index_load_path_ + "_pq_pivots.bin";
    std::string pq_coords_path = this->index_load_path_ + "_pq_compressed.bin";

    /* 1. Load PQ compressed vectors */
    uint32_t pq_num_centroids, pq_pivots_dim;
    uint32_t pq_num_pts;
    diskann::get_bin_metadata(pq_coords_path, pq_num_pts, pq_pivots_dim);
    std::cout << "PQ pivots metadata: pq_num_pts = " << pq_num_pts
              << ", dim = " << pq_pivots_dim << std::endl;
    if (pq_num_pts != this->num_points_) {
      std::cout << "ERROR: PQ num points (" << pq_num_pts
                << ") != index num points (" << this->num_points_ << ")"
                << std::endl;
      exit(1);
    }
    this->pq_dim_ = pq_pivots_dim;
    // 64-bit alignment
    this->pq_aligned_dim_ = ROUND_UP(this->pq_dim_, 8);
    uint64_t pq_data_size = this->num_points_ * this->pq_aligned_dim_;
    pq_data_size = ROUND_UP(pq_data_size, 256);
    // alloc aligned mem for PQ data
    this->pq_data_ = (uint8_t *)FASTER::core::aligned_alloc(256, pq_data_size);
    std::fill(this->pq_data_, this->pq_data_ + pq_data_size, 0);
    diskann::populate_from_bin<uint8_t>(this->pq_data_, pq_coords_path,
                                        this->num_points_, this->pq_dim_,
                                        this->pq_aligned_dim_);

    /* 2. Load PQ pivots */
    // load centroids
    this->pq_table.load_pq_centroid_bin(pq_pivots_path.c_str(), this->pq_dim_);
  }

  ~FasterDiskANNIndex() {
    if (this->graph_ != nullptr) {
      // std::cout << "~FasterDiskANNIndex::Freeing FASTER store object " <<
      // std::endl;
      delete this->graph_;
    }
    if (this->pq_data_ != nullptr) {
      // std::cout << "~FasterDiskANNIndex::Freeing PQ data" << std::endl;
      FASTER::core::aligned_free(this->pq_data_);
    }
    if (this->cached_node_data_buf != nullptr) {
      // std::cout << "~FasterDiskANNIndex::Freeing cached node data" <<
      // std::endl;
      FASTER::core::aligned_free(this->cached_node_data_buf);
    }
    if (this->cached_node_nbrs_buf != nullptr) {
      // std::cout << "~FasterDiskANNIndex::Freeing cached node nbrs" <<
      // std::endl;
      FASTER::core::aligned_free(this->cached_node_nbrs_buf);
    }
  }

  // not tracking any thread ID for now
  void StartSession() { this->graph_->StartSession(); }

  void StopSession() { this->graph_->StopSession(); }

  // cache BFS levels 0-N for the index
  void cache_bfs_levels(uint32_t max_num_nodes) {
    // allocate memory for cached node data
    this->cached_node_data_size = this->aligned_dim_ * sizeof(float);
    this->cached_node_data_size = ROUND_UP(this->cached_node_data_size, 64);
    uint64_t cached_node_data_buf_size =
        max_num_nodes * this->cached_node_data_size;
    this->cached_node_data_buf =
        (float *)FASTER::core::aligned_alloc(64, cached_node_data_buf_size);
    // allocate memory for cached node nbrs
    this->cached_node_nbrs_size = this->max_degree_ * sizeof(uint32_t);
    this->cached_node_nbrs_size = ROUND_UP(this->cached_node_nbrs_size, 64);
    uint64_t cached_node_nbrs_buf_size =
        max_num_nodes * this->cached_node_nbrs_size;
    this->cached_node_nbrs_buf =
        (uint32_t *)FASTER::core::aligned_alloc(64, cached_node_nbrs_buf_size);

    // zero both arrays
    std::fill(this->cached_node_data_buf,
              this->cached_node_data_buf + cached_node_data_buf_size, 0);
    std::fill(this->cached_node_nbrs_buf,
              this->cached_node_nbrs_buf + cached_node_nbrs_buf_size, 0);

    uint32_t cur_node_id = this->start_;
    uint32_t cur_map_idx = 0;
    std::queue<uint32_t> bfs_queue;
    bfs_queue.push(cur_node_id);
    while (!bfs_queue.empty() and cur_map_idx < max_num_nodes) {
      // get current node ID to read from FASTER store
      cur_node_id = bfs_queue.front();
      bfs_queue.pop();
      float *cur_node_data = this->cached_node_data_buf +
                             (cur_map_idx * this->cached_node_data_size);
      uint32_t *cur_node_nbrs = this->cached_node_nbrs_buf +
                                (cur_map_idx * this->cached_node_nbrs_size);
      // read node data from FASTER store
      uint32_t vec_dim = 0, num_nbrs = 0;
      this->Read(cur_node_id, cur_node_data, vec_dim, cur_node_nbrs, num_nbrs);
      if (vec_dim != this->dim_) {
        std::cout << "ERROR: vector dimension mismatch for node " << cur_node_id
                  << std::endl;
        exit(1);
      }
      if (num_nbrs > this->max_degree_) {
        std::cout << "ERROR: number of neighbors exceeds max degree for node "
                  << cur_node_id << std::endl;
        exit(1);
      }
      // add neighbors to queue
      for (uint32_t i = 0; i < num_nbrs; i++) {
        bfs_queue.push(cur_node_nbrs[i]);
      }
      // add to map
      this->cached_node_ids_[cur_node_id] =
          std::make_pair(cur_map_idx, num_nbrs);
      // increment map index
      cur_map_idx++;
    }
    std::cout << "Cached " << cur_map_idx << " nodes" << std::endl;
  }

  /*** Upsert ***/
  // Upsert (blindly replace without reading) adjacency list for a node
  // args: node_id: node ID (uint32_t)
  //       data: vector data (float*)
  //       dim: dimension of vector (uint32_t)
  //       nbrs: array of neighbors (uint32_t*)
  //       num_nbrs: number of neighbors (uint32_t)
  // Ensure `nbrs` has at least `num_nbrs` elements
  // Ensure `data` has at least `dim` elements
  void Upsert(uint32_t node_id, float *data, uint32_t dim, uint32_t *nbrs,
              uint32_t num_nbrs) {
    auto callback = [](IAsyncContext *ctxt, Status result) {};
    // create upsert context
    diskann::DiskannUpsertContext<float> context{node_id, data, dim, nbrs,
                                                 num_nbrs};
    // upsert into store
    auto result = this->graph_->Upsert(context, callback, 1);
    if (result != Status::Ok) {
      std::cout << "Upsert failed with status " << result << std::endl;
    } else {
      // std::cout << "upsert nnbrs: " << num_nbrs << std::endl;
    }
  }

  /*** Read ***/
  // Read adjacency list for a node
  // args: node_id: node ID (uint32_t)
  //       data: vector data (float*)
  //       dim: dimension of vector (uint32_t)
  //       nbrs: array of neighbors (uint32_t*)
  //       num_nbrs: number of neighbors (uint32_t)
  // Ensure `nbrs` has space to write least `max_degree_` elements
  void Read(uint32_t node_id, float *data, uint32_t &dim, uint32_t *nbrs,
            uint32_t &num_nbrs) {
    // std::cout << "Reading key:" << node_id << std::endl;
    auto callback = [](IAsyncContext *ctxt, Status result) {};

    // create read context
    diskann::DiskannReadContext<float> context{node_id, data, &dim, nbrs,
                                               &num_nbrs};

    // read from store
    auto result = this->graph_->Read(context, callback, 1);
    if (result != Status::Ok) {
      std::cout << "Read " << node_id << ", " << nbrs << ", " << num_nbrs
                << " --> failed with status " << result << std::endl;
    }

    // std::cout << "Read " << num_nbrs << " neighbors for key " << node_id <<
    // std::endl;
  }

  /*** Query ***/
  // Query for k nearest neighbors
  // ensure L_search < PQ_DEFAULT_SIZE --> priority queue size
  void search(const float *query, const uint32_t k_NN, const uint32_t L_search,
              uint32_t *knn_idxs, float *knn_dists, QueryStats *query_stats,
              uint32_t beam_width = 4, QueryContext *context = nullptr) {
    // initialize priority queue of neighbors
    // unexplored_front --> PQ distances
    CloserPQ *unexplored_front = context->unexplored_front;
    // explored_front --> float (full-precision) distances
    CloserPQ *explored_front = context->explored_front;
    explored_front->trim(0);
    unexplored_front->trim(0);

    // set of visited nodes
    std::unordered_set<uint32_t> visited_set;

    /* PQ-related initial query processing */
    PQScratch<float> *pq_scratch = context->pq_scratch;
    pq_scratch->set(this->dim_, query);
    float *centered_query = pq_scratch->rotated_query;
    // pre-process rotated vector (center)
    // query <-> PQ chunk centers distances
    float *pq_dists = pq_scratch->aligned_pqtable_dist_scratch;
    // node nbrs PQ coord scratch (for locality of reference)
    uint8_t *pq_coord_scratch = pq_scratch->aligned_pq_coord_scratch;
    // query <-> node nbr dist scratch
    float *dist_scratch = pq_scratch->aligned_dist_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_pq_dists = [&, this, pq_coord_scratch, pq_dists,
                             dist_scratch](const uint32_t *input_ids,
                                           const uint64_t n_ids) {
      diskann::aggregate_coords(input_ids, n_ids, this->pq_data_, this->pq_dim_,
                                pq_coord_scratch, this->pq_aligned_dim_);
      diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->pq_dim_, pq_dists,
                              dist_scratch, this->pq_aligned_dim_);
      query_stats->n_cmps += n_ids;
    };

    // populate query <-> PQ chunk distances
    std::copy(query, query + this->dim_, centered_query);        // mutable
    this->pq_table.preprocess_query(centered_query);             // center
    pq_table.populate_chunk_distances(centered_query, pq_dists); // dists

    Candidate *cur_beam = context->cur_beam;
    Candidate *beam_new_cands = context->beam_new_cands;
    uint32_t beam_num_new_cands = 0;
    uint32_t cur_beam_size = 0;
    uint32_t **beam_nbrs = context->beam_nbrs;
    uint32_t *beam_nnbrs = context->beam_nnbrs;
    float **beam_nbrs_data = context->beam_nbrs_data;

    auto get_float_dist = [&, this, query](const float *node_vec) {
      query_stats->n_cmps++;
      return diskann::compare<float>(query, node_vec, this->aligned_dim_);
    };
    // lambda to read cur beam nodes from disk
    auto read_cur_beam_from_disk = [&, this, cur_beam, beam_nnbrs, beam_nbrs,
                                    beam_nbrs_data](uint32_t count) {
      // std::cout << "Reading " << count << " nodes from disk" << std::endl;
      uint64_t start_tsc = __builtin_ia32_rdtsc();
      uint64_t cur_io_size = 0;
      for (uint32_t k = 0; k < count; k++) {
        // std::cout << "Reading node " << cur_beam[k].id << " from disk" <<
        // std::endl;
        uint32_t &node_nnbrs = beam_nnbrs[k], node_vec_dim = 0;
        uint32_t *node_nbrs = beam_nbrs[k];
        float *node_vec = beam_nbrs_data[k];
        uint32_t node_idx = cur_beam[k].id;
        this->Read(node_idx, node_vec, node_vec_dim, node_nbrs, node_nnbrs);
        _mm_prefetch((const char *)node_nbrs, _MM_HINT_T1);
        _mm_prefetch((const char *)node_vec, _MM_HINT_T1);
        // replace PQ dist with float dist for beam node
        float new_dist = get_float_dist(node_vec);
        // std::cout << "Node ID: " << cur_beam[k].id << ", PQ dist: " <<
        // cur_beam[k].dist << ", FP dist: " << new_dist << std::endl;
        cur_beam[k].dist = new_dist;
        assert(node_vec_dim == this->dim_);
        assert(node_nnbrs <= this->max_degree_);
        cur_io_size += ((node_nnbrs) * sizeof(uint32_t) +
                        sizeof(diskann::DiskannValue<float>));
      }
      uint64_t end_tsc = __builtin_ia32_rdtsc();
      query_stats->io_ticks += (end_tsc - start_tsc);
      query_stats->n_ios += cur_beam_size;
      query_stats->read_size += cur_io_size;
    };

    /* seed search with start node */
    cur_beam[0] = Candidate{this->start_, 1e7f};
    visited_set.insert(this->start_);
    unexplored_front->push_batch(cur_beam, 1);

    uint32_t MAX_ITERS = 10;
    uint32_t cur_iter = query_stats->n_hops;
    query_stats->io_ticks = 0;
    // start query search
    while (unexplored_front->size() > 0 && cur_iter < MAX_ITERS) {
      // reset iter variables
      cur_iter++;
      cur_beam_size = 0;

      // test early convergence of greedy search: top unexplored is worse than
      // worst explored
      if (explored_front->size() > 0) {
        float worst_explored_dist = explored_front->worst().dist;
        float best_unexplored_dist = unexplored_front->best().dist;
        if (best_unexplored_dist > worst_explored_dist) {
          // std::cout << "Early convergence: " << best_unexplored_dist << " >
          // "<< worst_explored_dist << std::endl;
          break;
        }
      }

      // collect up-to `beam_width` closest candidates from unexplored front
      uint32_t pop_count = 0;
      const Candidate *unexplored_front_data = unexplored_front->data();
      for (uint32_t i = 0;
           i < unexplored_front->size() && cur_beam_size < beam_width; i++) {
        const Candidate &cand = unexplored_front_data[i];
        // record as popped
        pop_count++;
        // get current closest
        cur_beam[cur_beam_size] = cand;
        uint32_t cur_node_id = cand.id;
        float cur_node_dist = cand.dist;
        // std::cout << "[" << cur_beam_size << "]" << "Cur node : " <<
        // cur_node_id << "," << cur_node_dist<< std::endl; increment beam size
        cur_beam_size++;
      }

      // pop number of considered candidates
      unexplored_front->pop_best_n(pop_count);

      // read all beam nodes from disk
      read_cur_beam_from_disk(cur_beam_size);

      // std::cout << "Iter: " << cur_iter << ", beam size: " << cur_beam_size
      // << std::endl;
      for (uint32_t i = 0; i < cur_beam_size; i++) {
        // std::cout << "cand: " << cur_beam[i].id << ", " << cur_beam[i].dist
        // << std::endl;
      }

      // number of viable new candidates generated
      beam_num_new_cands = 0;
      // iterate over neighbors for each candidate
      for (uint32_t i = 0; i < cur_beam_size; i++) {
        // get candidate
        Candidate &cur_cand = cur_beam[i];
        // get neighbors
        uint32_t *nbrs = beam_nbrs[i];
        uint32_t num_nbrs = beam_nnbrs[i];
        // compute dists for (centerd_query -> nbrs of cur_cand)
        compute_pq_dists(nbrs, num_nbrs);
        // iterate over neighbors for this candidate
        for (uint32_t j = 0; j < num_nbrs; j++) {
          // get neighbor ID
          uint32_t nbr_id = nbrs[j];
          // check if neighbor is in visited set
          if (visited_set.find(nbr_id) != visited_set.end()) {
            // skip neighbor
            continue;
          }
          // compute distance to query
          float nbr_dist = dist_scratch[j];
          // std::cout << "Neighbor: " << nbr_id << ", dist: " << nbr_dist <<
          // std::endl; check if `nbr_id` dist is worse than worst in explored
          // front
          if (explored_front->size() > 0) {
            const Candidate &worst_ex = explored_front->worst();
            if (nbr_dist >= worst_ex.dist) {
              // skip `nbr_id` if worse than worst in explored front
              // std::cout << "Skipping: " << nbr_id << ", dist: " << nbr_dist
              // << std::endl;
              continue;
            }
          }

          // collect candidate for insertion
          beam_new_cands[beam_num_new_cands++] = Candidate{nbr_id, nbr_dist};

          // mark visited
          visited_set.insert(nbr_id);
        }
      }

      // insert all collected candidates into unexplored front
      // std::cout << "Collected " << beam_num_new_cands << " candidates to
      // insert into unexplored front." << std::endl;
      for (uint32_t i = 0; i < beam_num_new_cands; i++) {
        // std::cout << "Queueing: " << beam_new_cands[i].id << ", dist: " <<
        // beam_new_cands[i].dist << std::endl;
      }
      unexplored_front->push_batch(beam_new_cands, beam_num_new_cands);
      unexplored_front->trim(L_search);

      // add `beam` to explored front, truncate to best L_search
      // std::cout << "Pushing " << cur_beam_size << " candidates to explored
      // front." << std::endl;
      explored_front->push_batch(cur_beam, cur_beam_size);
      explored_front->trim(L_search);
    }

    // record num iters
    query_stats->n_hops = cur_iter;
    // copy results to output
    // std::cout << std::endl << "Explored front:";
    const Candidate *explored_front_data = explored_front->data();
    for (uint32_t i = 0; i < explored_front->size(); i++) {
      if (i < k_NN) {
        knn_idxs[i] = explored_front_data[i].id;
        knn_dists[i] = explored_front_data[i].dist;
      }
      // std::cout << "( " << explored_front_data[i].id << ", " <<
      // explored_front_data[i].dist << " ), ";
    }
    // std::cout << std::endl;
    // std::cout << "Visited " << visited_set.size() << " nodes:";
    // print all visited IDs
    for (auto it = visited_set.begin(); it != visited_set.end(); it++) {
      // std::cout << *it << ", ";
    }
    // std::cout << std::endl;
  }

  // quick in-memory search over 1-10k nodes to determine
  // best 5-10 nodes to expand
  void _blitz_search(const float *query, const uint32_t k_NN,
                     const uint32_t L_search, QueryStats *query_stats,
                     QueryContext *context,
                     std::unordered_set<uint32_t> &visited_set) {
    // default value for _blitz_search
    uint32_t beam_width = 1;
    // priority queue of neighbors
    CloserPQ *unexplored_front = context->unexplored_front;
    CloserPQ *explored_front = context->explored_front;

    // cached neighbor list
    Candidate *cur_beam = context->cur_beam;
    Candidate *beam_new_cands = context->beam_new_cands;
    uint32_t beam_num_new_cands = 0;
    uint32_t cur_beam_size = 0;
    uint32_t **beam_nbrs = context->beam_nbrs;
    uint32_t *beam_nnbrs = context->beam_nnbrs;

    // lambda to read a node nbrs
    auto get_cached_nbrs = [&, this](uint32_t node_idx, uint32_t *&node_nbrs,
                                     uint32_t &node_nnbrs) {
      uint32_t map_idx = this->cached_node_ids_[node_idx].first;
      node_nnbrs = this->cached_node_ids_[node_idx].second;
      this->cached_node_data_buf + (map_idx * this->cached_node_data_size);
      node_nbrs =
          this->cached_node_nbrs_buf + (map_idx * this->cached_node_nbrs_size);
      query_stats->n_cache++;
    };

    auto get_cached_query_dist = [&, this](const float *query,
                                           const uint32_t node_id) {
      uint32_t map_idx = this->cached_node_ids_[node_id].first;
      const float *node_vec =
          this->cached_node_data_buf + (map_idx * this->cached_node_data_size);
      query_stats->n_cmps++;
      return diskann::compare<float>(query, node_vec, this->aligned_dim_);
    };

    // seed `unexplored_front` with start node
    uint32_t start_node_idx = this->start_;
    float query_start_dist = get_cached_query_dist(query, start_node_idx);
    beam_new_cands[0] = Candidate{start_node_idx, query_start_dist};
    unexplored_front->push_batch(beam_new_cands, 1);
    visited_set.insert(start_node_idx);

    uint32_t MAX_ITERS = 1000, cur_iter = 0;
    query_stats->io_ticks = 0;
    // start query search
    while (unexplored_front->size() > 0 && cur_iter < MAX_ITERS) {
      // reset iter variables
      cur_iter++;
      cur_beam_size = 0;

      // test early convergence of greedy search: top unexplored is worse than
      // worst explored
      if (explored_front->size() > 0) {
        if (unexplored_front->best().dist > explored_front->worst().dist)
          break;
      }

      // populate `beam_width` closest candidates from unexplored front
      uint32_t pop_count = 0;
      const Candidate *unexplored_front_data = unexplored_front->data();
      for (uint32_t i = 0; i < unexplored_front->size(); i++) {
        const Candidate &cand = unexplored_front_data[i];
        if (cur_beam_size >= beam_width)
          break;
        // record as popped
        pop_count++;
        // get current closest
        cur_beam[cur_beam_size] = cand;
        uint32_t cur_node_id = cand.id;
        float cur_node_dist = cand.dist;
        // std::cout << "[" << cur_beam_size << "]" << "Cur node : " <<
        // cur_node_id << "," << cur_node_dist << std::endl;

        // increment beam size
        cur_beam_size++;
      }

      // pop number of considered candidates
      unexplored_front->pop_best_n(pop_count);

      // std::cout << "Iter: " << cur_iter << ", beam size: " << cur_beam_size
      // << std::endl;
      for (uint32_t i = 0; i < cur_beam_size; i++) {
        // std::cout << "cand: " << cur_beam[i].id << ", " << cur_beam[i].dist
        // << std::endl;
      }

      // number of viable new candidates generated
      beam_num_new_cands = 0;
      // iterate over neighbors for each candidate
      for (uint32_t i = 0; i < cur_beam_size; i++) {
        // get candidate
        Candidate cur_cand = cur_beam[i];
        // get neighbors
        uint32_t *nbrs, num_nbrs;
        get_cached_nbrs(cur_cand.id, nbrs, num_nbrs);

        // iterate over neighbors for this candidate
        for (uint32_t i = 0; i < num_nbrs; i++) {
          // get neighbor ID
          uint32_t nbr_id = nbrs[i];
          // check if neighbor is in visited set
          if (visited_set.find(nbr_id) != visited_set.end()) {
            // skip neighbor
            continue;
          }
          // check if nbr is in cached set; skip if not
          auto iter = this->cached_node_ids_.find(nbr_id);
          if (iter == this->cached_node_ids_.end()) {
            continue;
          }

          // compute distance to query for nbr_id
          float nbr_dist = get_cached_query_dist(query, nbr_id);

          // check if `nbr_id` dist is worse than worst in explored front
          if (explored_front->size() > 0) {
            const Candidate &worst_ex = explored_front->worst();
            if (nbr_dist >= worst_ex.dist) {
              // skip `nbr_id` if worse than worst in explored front
              // std::cout << "Skipping: " << nbr_id << ", dist: " << nbr_dist
              // << std::endl;
              continue;
            }
          }

          // collect candidate for insertion
          beam_new_cands[beam_num_new_cands++] = Candidate{nbr_id, nbr_dist};

          // mark visited
          visited_set.insert(nbr_id);
        }
      }

      // insert all collected candidates into unexplored front
      for (uint32_t i = 0; i < beam_num_new_cands; i++) {
        // std::cout << "Queueing: " << beam_new_cands[i].id << ", dist: " <<
        // beam_new_cands[i].dist << std::endl;
      }
      unexplored_front->push_batch(beam_new_cands, beam_num_new_cands);
      unexplored_front->trim(L_search);

      // add `beam` to explored front, truncate to best L_search
      explored_front->push_batch(cur_beam, cur_beam_size);
      explored_front->trim(L_search);
    }

    // record num iters
    query_stats->n_hops = cur_iter;

    // copy results to output
    // std::cout << " _blitz_search(): unexplored_front ==>";
    const Candidate *unexplored_front_data = unexplored_front->data();
    for (uint32_t i = 0; i < unexplored_front->size(); i++) {
      // std::cout << "( " << unexplored_front_data[i].id << ", " <<
      // unexplored_front_data[i].dist << " ), ";
    }
    // std::cout << std::endl;
    // std::cout << "Visited " << visited_set.size() << " nodes:";
    // print all visited IDs
    for (auto it = visited_set.begin(); it != visited_set.end(); it++) {
      // std::cout << *it << ", ";
    }
    // std::cout << std::endl;
  }
};
} // namespace diskann