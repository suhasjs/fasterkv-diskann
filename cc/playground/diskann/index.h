// Implementation of a Vamana index (purely in-memory, using a FASTER NullDisk)
// Author: Suhas Jayaram Subramanya (suhasj@cs.cmu.edu)

#pragma once
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_set>

#include "graph.h"
#include "io_utils.h"
#include "search_utils.h"

#include "../src/core/faster.h"

#define MIN_FASTER_LOG_SIZE (1 << 30)  // 1 GB
#define FASTER_LOG_ALIGNMENT (1 << 25) // 32 MB

namespace diskann {
class FasterVamanaIndex {
private:
  // basic index params
  uint64_t num_points_ = 0, dim_ = 0, aligned_dim_ = 0;
  uint64_t max_degree_ = 0;
  diskann::MemGraph *graph_ = nullptr;
  float *data_ = nullptr;
  uint32_t start_ = 0;

  // index parameters
  std::string index_load_path_ = "";

  // FASTER store params
  uint64_t faster_max_keys_ = 0;
  uint64_t faster_memory_size_ = 0;
  std::string faster_graph_path_ = "";

  // extra config params
  // whether to verify after loading graph data
  bool verify_after_load_ = false;

public:
  FasterVamanaIndex(const uint64_t num_points, const uint64_t dim,
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
        sizeof(diskann::FlexibleValue) +          // size of object
        (sizeof(uint32_t) * (approx_degree + 1)); // size of neighbors list
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
    this->graph_ =
        new diskann::MemGraph(this->faster_max_keys_, this->faster_memory_size_,
                              this->faster_graph_path_);
    std::cout << "Created FASTER store for VamanaIndex" << std::endl;
    std::cout << "Finished configuring index. Call load() to load graph into "
                 "FASTER store."
              << std::endl;
  }

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
      this->Upsert(node_id, nbrs, num_nbrs);

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
        this->Read(node_id, nbrs2, num_nbrs2);
        // verify
        if (num_nbrs != num_nbrs2) {
          std::cout << "ERROR: num_nbrs mismatch for node " << node_id
                    << " (disk = " << num_nbrs << ", FASTER = " << num_nbrs2
                    << ")" << std::endl;
          exit(1);
        }
        for (uint32_t j = 0; j < num_nbrs; j++) {
          if (nbrs[j] != nbrs2[j]) {
            std::cout << "ERROR: nbrs mismatch for node " << node_id
                      << " (disk = " << nbrs[j] << ", FASTER = " << nbrs2[j]
                      << ")" << std::endl;
            exit(1);
          }
        }
      }
      delete[] nbrs2;
    }
    delete[] nbrs;

    std::cout << "Finished loading graph data and verifying insertion into "
                 "FASTER store"
              << std::endl;

    // close graph file
    graph_reader.close();
  }

  ~FasterVamanaIndex() {
    if (this->data_ != nullptr) {
      std::cout << "Freeing vector data " << std::endl;
      FASTER::core::aligned_free(this->data_);
    }
    if (this->graph_ != nullptr) {
      std::cout << "Freeing FASTER store object " << std::endl;
      delete this->graph_;
    }
  }

  // not tracking any thread ID for now
  void StartSession() { this->graph_->StartSession(); }

  void StopSession() { this->graph_->StopSession(); }

  /*** Upsert ***/
  // Upsert (blindly replace without reading) adjacency list for a node
  // args: node_id: node ID (uint32_t)
  //       nbrs: array of neighbors (uint32_t*)
  //       num_nbrs: number of neighbors (uint32_t)
  // Ensure `nbrs` has at least `num_nbrs` elements
  void Upsert(uint32_t node_id, uint32_t *nbrs, uint32_t num_nbrs) {
    auto callback = [](IAsyncContext *ctxt, Status result) {
      std::cout
          << "Upsert completed for key: "
          << reinterpret_cast<diskann::GraphUpsertContext *>(ctxt)->key().key
          << std::endl;
    };
    // create upsert context
    diskann::GraphUpsertContext context{node_id, nbrs, num_nbrs};
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
  //       nbrs: array of neighbors (uint32_t*)
  //       num_nbrs: number of neighbors (uint32_t)
  // Ensure `nbrs` has space to write least `max_degree_` elements
  void Read(uint32_t node_id, uint32_t *nbrs, uint32_t &num_nbrs) {
    // std::cout << "Reading key:" << node_id << std::endl;
    auto callback = [](IAsyncContext *ctxt, Status result) {
      std::cout
          << "Read completed for key: "
          << reinterpret_cast<diskann::GraphReadContext *>(ctxt)->key().key
          << std::endl;
    };

    // create read context
    diskann::GraphReadContext context{node_id, nbrs, &num_nbrs};

    // read from store
    auto result = this->graph_->Read(context, callback, 1);
    if (result != Status::Ok) {
      std::cout << "Read failed with status " << result << std::endl;
    }

    // std::cout << "Read " << num_nbrs << " neighbors for key " << node_id <<
    // std::endl;
  }

  /*** Query ***/
  // Query for k nearest neighbors
  // ensure L_search < PQ_DEFAULT_SIZE --> priority queue size
  void search(const float *query, const uint32_t k_NN, const uint32_t L_search,
              uint32_t *knn_idxs, float *knn_dists) {
    // initialize priority queue of neighbors
    CloserPQ unexplored_front, explored_front;

    // initialize visited set
    std::unordered_set<uint32_t> visited_set;

    // seed with start node
    uint32_t start_node_idx = this->start_;
    const float *start_node_vec =
        this->data_ + (start_node_idx * this->aligned_dim_);
    float query_start_dist =
        diskann::compare<float>(query, start_node_vec, this->aligned_dim_);
    Candidate start_cand{start_node_idx, query_start_dist};
    unexplored_front.push(start_cand);

    // cached neighbor list
    uint32_t nbrs_cache[128];
    Candidate temp_cands[128];
    uint32_t *nbrs = &nbrs_cache[0];
    uint32_t MAX_ITERS = 20000, cur_iter = 0;

    // start query search
    while (unexplored_front.size() > 0 && cur_iter < MAX_ITERS) {
      cur_iter++;
      // pull closest vertex from unexplored front
      Candidate top_unex = unexplored_front.pop_best();
      uint32_t cur_node_id = top_unex.id;
      float cur_node_dist = top_unex.dist;
      // skip if already visited
      if (visited_set.find(cur_node_id) != visited_set.end()) {
        continue;
      }

      // read neighbors of candidate
      uint32_t num_nbrs = 0;
      this->Read(top_unex.id, nbrs, num_nbrs);
      // std::cout << "[" << unexplored_front.size() << "]" << "Cur node : " <<
      // cur_node_id << "," << cur_node_dist << << num_nbrs << std::endl;

      // iterate over neighbors
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
        // compute distance to query
        float nbr_dist =
            diskann::compare<float>(query, nbr_vec, this->aligned_dim_);
        // check if `nbr_id` dist is worse than worst in explored front
        if (explored_front.size() == k_NN) {
          const Candidate &worst_ex = explored_front.worst();
          if (nbr_dist >= worst_ex.dist) {
            // skip `nbr_id` if worse than worst in explored front
            continue;
          }
        }

        // create candidate
        Candidate nbr_cand{nbr_id, nbr_dist};
        // add to unexplored front
        unexplored_front.push(nbr_cand);
        // std::cout << "Queueing: " << nbr_id << ", dist: " << nbr_dist <<
        // std::endl;
        // trim worst in unexplored front if needed
        if (unexplored_front.size() > L_search) {
          unexplored_front.pop_worst();
        }
      }

      // add `top_unex` to explored front, truncate to best k_NN
      explored_front.push(top_unex);
      if (explored_front.size() > k_NN) {
        explored_front.pop_worst();
      }
      // mark `top_unex` as visited
      visited_set.insert(top_unex.id);
    }

    // copy results to output
    // std::cout << "Final results:";
    for (uint32_t i = 0; i < k_NN; i++) {
      Candidate ith_NN = explored_front.pop_best();
      knn_idxs[i] = ith_NN.id;
      knn_dists[i] = ith_NN.dist;
      // std::cout << "( " << ith_NN.id << ", " << ith_NN.dist << " ), ";
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