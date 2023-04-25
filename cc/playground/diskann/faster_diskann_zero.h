// Implementation of a zero-additional memory DiskANN index (purely on-disk,
// using a FASTER FilesystemDisk)
// Author: Suhas Jayaram Subramanya (suhasj@cs.cmu.edu)

#pragma once
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_set>

#include "disk_graph.h"
#include "io_utils.h"
#include "search_utils.h"
#include <immintrin.h>

#include "../src/core/faster.h"

#define MIN_FASTER_LOG_SIZE (1 << 30)  // 1 GB
#define FASTER_LOG_ALIGNMENT (1 << 25) // 32 MB
#define MAX_BEAM_WIDTH (1 << 16)       // max beam width for beam search
#define MAX_VAMANA_DEGREE (1 << 7)     // max degree for Vamana graph

namespace diskann {
class FasterDiskANNZeroIndex {
private:
  // basic index params
  uint64_t num_points_ = 0, dim_ = 0, aligned_dim_ = 0;
  uint64_t max_degree_ = 0;
  diskann::DiskGraph *graph_ = nullptr;
  float *data_ = nullptr;
  uint32_t start_ = 0;

  // index parameters
  std::string index_load_path_ = "";

  // FASTER store params
  uint64_t faster_max_keys_ = 0;
  uint64_t faster_memory_size_ = 0;
  std::string faster_graph_path_ = "/dev/shm/diskann_zero.store";

  // extra config params
  // whether to verify after loading graph data
  bool verify_after_load_ = true;

public:
  FasterDiskANNZeroIndex(const uint64_t num_points, const uint64_t dim,
                         const std::string &index_load_path)
      : num_points_{num_points}, dim_{dim}, index_load_path_{index_load_path} {
    /*** 1. Load vector data ****/
    std::string data_path = index_load_path + ".data";
    // round up dim to multiple of 16 (good alignment for AVX ops)
    this->aligned_dim_ = ROUND_UP(dim, 16);
    std::cout << "Allocating memory for vector data " << std::endl;
    this->data_ = reinterpret_cast<float *>(FASTER::core::aligned_alloc(
        1024, this->num_points_ * this->aligned_dim_ * sizeof(float)));
    // zero out the data array (to set unused dimensions to 0)
    memset(this->data_, 0, this->num_points_ * this->dim_ * sizeof(float));
    // populate vector data from data file
    diskann::populate_from_bin<float>(this->data_, data_path, this->num_points_,
                                      this->dim_, this->aligned_dim_);

    /*** 2. Configure FASTER store ****/
    // set max number of keys to be nearest power of 2 >= this->num_points_
    this->faster_max_keys_ = 1;
    while (this->faster_max_keys_ < this->num_points_)
      this->faster_max_keys_ <<= 1;
    std::cout << "Setting FASTER store max keys to " << this->faster_max_keys_
              << " (nearest power of 2 >= " << this->num_points_ << ")"
              << std::endl;
    // read graph metadata from disk
    size_t graph_filesize;
    diskann::get_graph_metadata(index_load_path, graph_filesize, this->start_);
    uint64_t approx_degree =
        ((graph_filesize / sizeof(uint32_t)) / this->num_points_) - 1;
    // compute memory requirement for FASTER store
    uint64_t per_key_memory =
        sizeof(diskann::DiskannValue<float>) +     // size of object
        (sizeof(uint32_t) * (approx_degree + 1)) + // size of neighbors list
        sizeof(float) * this->dim_;                // size of vector
    per_key_memory += sizeof(diskann::FixedSizeKey<uint32_t>); // size of key
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
              << " MB, per key memory = " << per_key_memory << std::endl;
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
    std::ifstream graph_reader(this->index_load_path_,
                               std::ios::in | std::ios::binary);
    // 24-byte header for Vamana graph
    graph_reader.seekg(24, std::ios::beg);
    // read and upsert graph data
    uint32_t node_id, num_nbrs;
    uint32_t *nbrs = new uint32_t[128];
    for (uint32_t i = 0; i < this->num_points_; i++) {
      node_id = i;
      // read num neighbors
      graph_reader.read(reinterpret_cast<char *>(&num_nbrs), sizeof(uint32_t));
      // read all neighbros
      graph_reader.read(reinterpret_cast<char *>(nbrs),
                        sizeof(uint32_t) * num_nbrs);
      // upsert into FASTER store
      float *node_data = this->data_ + (i * this->aligned_dim_);
      this->Upsert(node_id, node_data, this->dim_, nbrs, num_nbrs);

      // update max degree
      this->max_degree_ = std::max(this->max_degree_, (uint64_t)num_nbrs);
    }

    // round up max degree to nearest multiple of 8
    this->max_degree_ = ROUND_UP(this->max_degree_, 8);
    std::cout << "Max degree = " << this->max_degree_ << std::endl;

    // go back and verify all inserted data
    if (this->verify_after_load_) {
      std::cout << "Verifying inserted data" << std::endl;
      uint32_t num_nbrs2, *nbrs2 = new uint32_t[128];
      float *data2 = new float[this->aligned_dim_];
      graph_reader.close();
      graph_reader.open(this->index_load_path_,
                        std::ios::in | std::ios::binary);
      graph_reader.seekg(24, std::ios::beg);
      // read graph data
      for (uint32_t i = 0; i < this->num_points_; i++) {
        node_id = i;
        // read num neighbors from disk
        graph_reader.read(reinterpret_cast<char *>(&num_nbrs),
                          sizeof(uint32_t));
        // read all neighbros from disk
        graph_reader.read(reinterpret_cast<char *>(nbrs),
                          sizeof(uint32_t) * num_nbrs);
        // read both from FASTER store
        uint32_t read_dim = 0;
        this->Read(node_id, data2, read_dim, nbrs2, num_nbrs2);
        // verify metadata for node
        if (num_nbrs != num_nbrs2) {
          std::cout << "ERROR: num_nbrs mismatch for node " << node_id
                    << " (disk = " << num_nbrs << ", FASTER = " << num_nbrs2
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
          if (nbrs[j] != nbrs2[j]) {
            std::cout << "ERROR: nbrs mismatch for node " << node_id
                      << " (disk = " << nbrs[j] << ", FASTER = " << nbrs2[j]
                      << ")" << std::endl;
            exit(1);
          }
        }
        // verify data
        float *src_data = this->data_ + (i * this->aligned_dim_);
        if (memcmp(src_data, data2, this->dim_ * sizeof(float)) != 0) {
          std::cout << "ERROR: data mismatch for node " << node_id << std::endl;
          // print out mem data and disk data for debugging
          std::cout << "Mem vector = ";
          for (uint32_t j = 0; j < this->dim_; j++) {
            std::cout << src_data[j] << ",";
          }
          std::cout << std::endl;
          std::cout << "Disk vector = ";
          for (uint32_t j = 0; j < this->dim_; j++) {
            std::cout << data2[j] << ",";
          }
          std::cout << std::endl;
          exit(1);
        }
      }
      // verification successful
      std::cout << "Verification successful" << std::endl;
      delete[] nbrs2;
      delete[] data2;
    }
    delete[] nbrs;

    std::cout << "Finished loading graph data and verifying insertion into "
                 "FASTER store"
              << std::endl;

    // close graph file
    graph_reader.close();
  }

  ~FasterDiskANNZeroIndex() {
    if (this->data_ != nullptr) {
      // std::cout << "~FasterDiskANNZeroIndex::Freeing vector data " <<
      // std::endl;
      FASTER::core::aligned_free(this->data_);
    }
    if (this->graph_ != nullptr) {
      // std::cout << "~FasterDiskANNZeroIndex::Freeing FASTER store object " <<
      // std::endl;
      delete this->graph_;
    }
  }

  // not tracking any thread ID for now
  void StartSession() { this->graph_->StartSession(); }

  void StopSession() { this->graph_->StopSession(); }

  /*** Upsert ***/
  // Upsert (blindly replace without reading) adjacency list for a node
  // args: node_id: node ID (uint32_t)
  //       data: vector data (float*)
  //       dim: dimension of vector (uint32_t)
  //       nbrs: array of neighbors (uint32_t*)
  //       num_nbrs: number of neighbors (uint32_t)
  // Ensure `nbrs` has at least `num_nbrs` elements
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
    CloserPQ *unexplored_front = context->unexplored_front;
    CloserPQ *explored_front = context->explored_front;

    // initialize visited set
    std::unordered_set<uint32_t> visited_set;

    // cached neighbor list
    Candidate *cur_beam = context->cur_beam;
    Candidate *beam_new_cands = context->beam_new_cands;
    uint32_t beam_num_new_cands = 0;
    uint32_t cur_beam_size = 0;
    uint32_t **beam_nbrs = context->beam_nbrs;
    uint32_t *beam_nnbrs = context->beam_nnbrs;

    // seed `unexplored_front` with start node
    uint32_t start_node_idx = this->start_;
    const float *start_node_vec =
        this->data_ + (start_node_idx * this->aligned_dim_);
    // prefetch start node vector
    _mm_prefetch((const char *)start_node_vec, _MM_HINT_NTA);
    float query_start_dist =
        diskann::compare<float>(query, start_node_vec, this->aligned_dim_);
    query_stats->n_cmps++;
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

        // read neighbors of candidate
        uint32_t &num_nbrs = beam_nnbrs[cur_beam_size];
        // time IO
        {
          uint64_t start_tsc = __builtin_ia32_rdtsc();
          float *bad_data = nullptr;
          uint32_t bad_dim = 0;
          this->Read(cand.id, bad_data, bad_dim, beam_nbrs[cur_beam_size],
                     num_nbrs);
          query_stats->io_ticks += (__builtin_ia32_rdtsc() - start_tsc);
          _mm_prefetch((const char *)beam_nbrs[cur_beam_size], _MM_HINT_T0);
        }
        // record IO stats
        query_stats->n_ios++;
        query_stats->read_size += ((num_nbrs) * sizeof(uint32_t) +
                                   sizeof(diskann::DiskannValue<float>));
        assert(num_nbrs <= MAX_VAMANA_DEGREE);
        // std::cout << "[" << cur_beam_size << "]" << "Cur node : " <<
        // cur_node_id << "," << cur_node_dist << ", " << num_nbrs << std::endl;

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
        uint32_t *nbrs = beam_nbrs[i];
        uint32_t num_nbrs = beam_nnbrs[i];
        // iterate over neighbors for this candidate
        for (uint32_t i = 0; i < num_nbrs; i++) {
          // get neighbor ID
          uint32_t nbr_id = nbrs[i];
          // check if neighbor is in visited set
          if (visited_set.find(nbr_id) != visited_set.end()) {
            // skip neighbor
            continue;
          }
          // get vec for the neighbor
          const float *nbr_vec = this->data_ + (nbr_id * this->aligned_dim_);
          // prefetch neighbor vector
          _mm_prefetch((const char *)nbr_vec, _MM_HINT_NTA);
          // compute distance to query
          float nbr_dist =
              diskann::compare<float>(query, nbr_vec, this->aligned_dim_);
          query_stats->n_cmps++;
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
    // std::cout << "Final results:";
    const Candidate *explored_front_data = explored_front->data();
    for (uint32_t i = 0; i < k_NN; i++) {
      knn_idxs[i] = explored_front_data[i].id;
      knn_dists[i] = explored_front_data[i].dist;
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
};
} // namespace diskann