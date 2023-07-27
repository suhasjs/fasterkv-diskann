// Implementation of a Fresh-Vamana index (purely in-memory, using a FASTER NullDisk)
// Author: Suhas Jayaram Subramanya (suhasj@cs.cmu.edu)
// Faster logic borrowed from in_memory_test.cc:451 -- UpsertRead_ResizeValue_Concurrent

#pragma once
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>

#include "io_utils.h"
#include "mutable_mem_graph.h"
#include "search_utils.h"
#include <immintrin.h>

#include "../src/core/faster.h"

#define MIN_FASTER_LOG_SIZE (1 << 28)  // 256 MB
#define FASTER_LOG_ALIGNMENT (1 << 25) // 32 MB
#define MAX_BEAM_WIDTH (1 << 6)       // max beam width for beam search = 64
#define MAX_VAMANA_DEGREE (1 << 7)     // max degree for Vamana graph = 128

namespace diskann {
class FasterFreshVamanaIndex {
private:
  // basic index params
  uint64_t dim_ = 0, aligned_dim_ = 0;
  union {
    uint64_t num_points_;
    std::atomic<uint64_t> num_points_atomic_;
  };

  // max number of points supported by FreshVamanaIndex
  uint64_t max_num_points_ = 0;
  uint64_t max_degree_ = 0;

  // Faster store object --> mutable graph
  diskann::MutableGraph *graph_ = nullptr;
  // vector data stored as raw pointers
  float *data_ = nullptr;
  // start node ID for graph search
  uint32_t start_ = 0;
  // last active ID for graph search
  // bump this +1 for each insert
  // TODO :: deal with holes in active ID space
  std::atomic<uint64_t> last_active_id_{0};

  // index parameters
  std::string index_load_path_ = "";

  // FASTER store params
  uint64_t faster_max_keys_ = 0;
  uint64_t faster_memory_size_ = 0;
  std::string faster_graph_path_ = "";

  // extra config params
  // whether to verify after loading graph data
  bool verify_after_load_ = true;

public:
  FasterFreshVamanaIndex(const uint64_t max_num_points, const uint64_t max_degree, 
                         const std::string &index_load_path) : max_num_points_{max_num_points}, 
                         max_degree_{max_degree}, index_load_path_{index_load_path} {
    /*** 1. Load vector data ****/
    this->max_num_points_ = ROUND_UP(this->max_num_points_, 1024);
    std::string data_path = index_load_path + ".data";
    uint32_t npts_u32, dim_u32;
    diskann::get_bin_metadata(data_path, npts_u32, dim_u32);
    this->num_points_ = npts_u32;
    this->dim_ = dim_u32;
    this->last_active_id_.store(this->num_points_ - 1);
    std::cout << "Found " << this->num_points_ << " vectors with "
              << this->dim_ << " dims file: " << data_path << std::endl;
    // round up dim to multiple of 16 (good alignment for AVX ops)
    this->aligned_dim_ = ROUND_UP(this->dim_, 16);
    std::cout << "Allocating memory for vector data " << std::endl;
    this->data_ = reinterpret_cast<float *>(FASTER::core::aligned_alloc(
        1024, this->max_num_points_ * this->aligned_dim_ * sizeof(float)));
    // zero out the data array (to set unused dimensions to 0)
    memset(this->data_, 0, this->max_num_points_ * this->aligned_dim_ * sizeof(float));
    // populate vector data from data file
    diskann::populate_from_bin<float>(this->data_, data_path, this->num_points_,
                                      this->dim_, this->aligned_dim_);

    /*** 2. Configure FASTER store ****/
    // set max number of keys to be nearest power of 2 >= this->num_points_
    this->faster_max_keys_ = 1;
    while (this->faster_max_keys_ <= this->max_num_points_)
      this->faster_max_keys_ <<= 1;
    std::cout << "Setting FASTER store max keys to " << this->faster_max_keys_
              << " (nearest power of 2 >= " << this->max_num_points_ << ")"
              << std::endl;
    
    // read graph metadata from disk
    size_t graph_filesize;
    diskann::get_graph_metadata(index_load_path, graph_filesize, this->start_);

    // compute memory requirement for FASTER store
    uint64_t per_key_memory =
        sizeof(diskann::MutableFlexibleValue<uint32_t>) + // size of object
        (sizeof(uint32_t) * (this->max_degree_ + 1));  // size of neighbors list
    per_key_memory += sizeof(diskann::FixedSizeKey<uint32_t>); // size of key
    per_key_memory = ROUND_UP(per_key_memory, 8);
    this->faster_memory_size_ = this->faster_max_keys_ * per_key_memory;

    // round up to nearest 32 MB region
    this->faster_memory_size_ =
        ROUND_UP(this->faster_memory_size_, FASTER_LOG_ALIGNMENT);
    if (this->faster_memory_size_ < MIN_FASTER_LOG_SIZE)
      this->faster_memory_size_ = MIN_FASTER_LOG_SIZE;
    std::cout << "Configuring FASTER store memory size to "
              << this->faster_memory_size_ / (1 << 20)
              << " MB, per key memory = " << per_key_memory << "B" << std::endl;

    /*** 3. Create FASTER store ****/
    this->graph_ =
        new diskann::MutableGraph(this->faster_max_keys_, this->faster_memory_size_,
                                  this->faster_graph_path_);
    std::cout << "Created FASTER store for VamanaIndex" << std::endl;
    // std::cout << "Finished configuring index. Call load() to load graph into
    // FASTER store. " << std::endl;
  }

  uint64_t get_max_degree() { return this->max_degree_; }
  uint64_t get_num_points() { return this->last_active_id_.load(); }
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
      if (num_nbrs > this->max_degree_) {
        std::cout << "ERROR: num_nbrs = " << num_nbrs
                  << " > max_degree_ = " << this->max_degree_ << std::endl;
        exit(1);
      }
      // upsert into FASTER store
      this->Upsert(node_id, nbrs, num_nbrs);
    }

    // go back and verify all inserted data
    if (this->verify_after_load_) {
      std::cout << "Verifying inserted data (verify_after_load_ is true)"
                << std::endl;
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
        uint64_t node_gen;
        this->Read(node_id, nbrs2, num_nbrs2, node_gen);
        // verify if node generation is 0 (inserted for the first time)
        if (node_gen != 0) {
          std::cout << "ERROR: node_gen mismatch for node " << node_id
                    << " (expected = 0, FASTER = " << node_gen << ")" << std::endl;
          exit(1);
        }
        // verify if num_nbrs match
        if (num_nbrs != num_nbrs2) {
          std::cout << "ERROR: num_nbrs mismatch for node " << node_id
                    << " (disk = " << num_nbrs << ", FASTER = " << num_nbrs2
                    << ")" << std::endl;
          exit(1);
        }
        // verify each neighbor ID for node `node_id`
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

  ~FasterFreshVamanaIndex() {
    if (this->data_ != nullptr) {
      // std::cout << "~FasterFreshVamanaIndex::Freeing vector data " << std::endl;
      FASTER::core::aligned_free(this->data_);
    }
    if (this->graph_ != nullptr) {
      // std::cout << "~FasterFreshVamanaIndex::Freeing FASTER store object " <<
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
  //       nbrs: array of neighbors (uint32_t*)
  //       num_nbrs: number of neighbors (uint32_t)
  //       node_gen: expected generation number of node in Faster (uint64_t)
  // Ensure `nbrs` has at least `num_nbrs` elements
  // returns true if successful, false otherwise (atomic semantics)
  // returns false if node_gen != node generation in Faster (updated by another thread)
  bool Upsert(uint32_t node_id, uint32_t *nbrs, uint32_t num_nbrs, 
              uint64_t node_gen = std::numeric_limits<uint64_t>::max()) {
    auto callback = [](IAsyncContext *ctxt, Status result) {};
    // create upsert context
    diskann::MutableUpsertContext context{node_id, nbrs, num_nbrs, node_gen};
    // upsert into store
    auto result = this->graph_->Upsert(context, callback, 1);
    return (result == Status::Ok);
  }

  /*** Read ***/
  // Read adjacency list for a node
  // args: node_id: node ID (uint32_t)
  //       nbrs: array of neighbors (uint32_t*)
  //       num_nbrs: number of neighbors (uint32_t)
  //       node_gen: generation number of node (uint64_t)
  // Ensure `nbrs` has space to write least `max_degree_` elements
  void Read(uint32_t node_id, uint32_t *nbrs, uint32_t &num_nbrs, uint64_t &node_gen) {
    // std::cout << "Reading key:" << node_id << std::endl;
    auto callback = [](IAsyncContext *ctxt, Status result) {};

    // create read context
    diskann::MutableReadContext context{node_id, nbrs, &num_nbrs, &node_gen};

    // read from store
    auto result = this->graph_->Read(context, callback, 1);
    if (result != Status::Ok) {
      std::cout << "Read " << node_id << ", " << nbrs << ", " << num_nbrs
                << " --> failed with status " << result << std::endl;
    }

    // std::cout << "Read " << num_nbrs << " neighbors for key " << node_id <<
    // std::endl;
  }

  // prunes list of candidates in `cands` to at-most `max_degree_` best candidates
  void prune_neighbors(const std::vector<std::pair<uint32_t, float>> &cands, std::vector<uint32_t> &pruned_list) {
    std::vector<float> occlude_factors(cands.size(), 0.0f);
    float alpha = 1.2f;
    // occlude logic
    float cur_alpha = 1;
    while (cur_alpha <= alpha && pruned_list.size() < this->max_degree_) {
      for(uint64_t i=0; i < cands.size() && pruned_list.size() < this->max_degree_; i++) {
        if (occlude_factors[i] > cur_alpha) {
          continue;
        }
        // Set the entry to float::max so that is not considered again
        float* i_vec = this->data_ + (this->aligned_dim_ * (uint64_t) cands[i].first);
        occlude_factors[i] = std::numeric_limits<float>::max();
        pruned_list.push_back(cands[i].first);

        // Update occlude factor for points from i+1 to end of list
        for (uint64_t k = i + 1; k < cands.size(); k++) {
          if (occlude_factors[k] > alpha)
            continue;
          float* k_vec = this->data_ + (this->aligned_dim_ * (uint64_t) cands[k].first);
          float djk = diskann::compare<float>(k_vec, i_vec, this->aligned_dim_);
          occlude_factors[k] =
              (djk == 0) ? std::numeric_limits<float>::max()
                          : std::max(occlude_factors[k], cands[k].second / djk);
        }
      }
      cur_alpha *= 1.2;
    }

    // saturate pruned list to max degree
    for (const auto &cand : cands) {
      const uint32_t node_id = cand.first;
      if (pruned_list.size() >= this->max_degree_)
        break;
      if ((std::find(pruned_list.begin(), pruned_list.end(), node_id) ==
            pruned_list.end()))
        pruned_list.push_back(node_id);
    }
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
        if (cur_beam_size > beam_width)
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
          uint64_t node_gen = 0;
          this->Read(cand.id, beam_nbrs[cur_beam_size], num_nbrs, node_gen);
          query_stats->io_ticks += (__builtin_ia32_rdtsc() - start_tsc);
          _mm_prefetch((const char *)beam_nbrs[cur_beam_size], _MM_HINT_T0);
        }
        // record IO stats
        query_stats->n_ios++;
        query_stats->read_size += ((num_nbrs) * sizeof(uint32_t) +
                                   sizeof(diskann::MutableFlexibleValue<uint32_t>));
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
        for (uint32_t j = 0; j < num_nbrs; j++) {
          // get neighbor ID
          uint32_t nbr_id = nbrs[j];
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

  // returns ID of inserted node
  uint64_t insert(const float* new_vec, const uint32_t L_index, const float alpha=1.2f, 
                  QueryStats *query_stats = nullptr,
                  QueryContext *ctx = nullptr) {
    uint64_t last_active = this->last_active_id_.fetch_add(1);
    uint64_t new_id = last_active + 1;
    // std::cout << "Inserting new node with ID " << new_id << std::endl;
    // copy vector data to array
    float *new_vec_data = this->data_ + (new_id * this->aligned_dim_);
    std::memcpy(new_vec_data, new_vec, this->dim_ * sizeof(float));

    // search for neighbors, return closest `L_index` neighbors
    std::vector<uint32_t> knn_idxs(ROUND_UP(L_index, 64), 0);
    std::vector<float> knn_dists(ROUND_UP(L_index, 64), 0);
    this->search(new_vec, L_index, L_index, knn_idxs.data(), knn_dists.data(), query_stats, 1, ctx);

    // alloc memory for new neighbors
    std::vector<std::pair<uint32_t, float>> cands(L_index);
    /*
    std::cout << "Found candidates : " << std::endl;
    for(uint64_t i=0; i < L_index; i++) {
      std::cout << "(" << i << "->" << knn_idxs[i] << ", " << knn_dists[i] << ") ";
      cands[i] = std::make_pair(knn_idxs[i], knn_dists[i]);
    }
    std::cout << std::endl;
    */

    std::vector<uint32_t> pruned_nbrs;
    pruned_nbrs.reserve(this->max_degree_);

    // prune neighbors
    this->prune_neighbors(cands, pruned_nbrs);
    /*
    std::cout << "Pruned candidates: ";
    for(uint64_t i=0; i < pruned_nbrs.size(); i++) {
      std::cout << pruned_nbrs[i] << " ";
    }
    std::cout << std::endl;
    */

    // upsert new_id into FASTER store
    this->Upsert(new_id, pruned_nbrs.data(), pruned_nbrs.size());

    // inter-insert links
    std::vector<uint32_t> nbr_nbrs(this->max_degree_ + 8);
    for(uint64_t i=0; i < pruned_nbrs.size(); i++) {
      bool retry = true;
      uint32_t nbr_id = pruned_nbrs[i];
      // (read -> modify -> upsert)
      while(retry) {
        // read neighbors of candidate
        uint32_t num_nbrs = 0;
        uint64_t nbr_gen = std::numeric_limits<uint64_t>::max();
        this->Read(nbr_id, nbr_nbrs.data(), num_nbrs, nbr_gen);

        // add new node to neighbor's neighbor list
        nbr_nbrs[num_nbrs++] = new_id;

        // trigger prune if num_nbrs > max_degree_
        if (num_nbrs > this->max_degree_) {
          /*
          std::cout << "Calling prune for node " << nbr_id << std::endl;
          std::cout << "Old nbr list: ";
          for(uint64_t k=0; k < num_nbrs; k++) {
            std::cout << nbr_nbrs[k] << " ";
          }
          std::cout << std::endl;
          */
          cands.resize(num_nbrs);
          for(uint64_t k=0;k < num_nbrs; k++) {
            float *src_vec = this->data_ + (nbr_id * this->aligned_dim_);
            float *dest_vec = this->data_ + (nbr_nbrs[k] * this->aligned_dim_);
            float src_dest_dist = diskann::compare<float>(src_vec, dest_vec, this->aligned_dim_);
            cands[k] = std::make_pair(nbr_nbrs[k], src_dest_dist);
          }
          nbr_nbrs.clear();
          // don't need to reserve as std::vector will grow only if it exceeds capacity
          // nbr_nbrs.reserve(this->max_degree_);

          // prune neighbors into nbr_nbrs
          this->prune_neighbors(cands, nbr_nbrs);
          num_nbrs = nbr_nbrs.size();
          /*
          std::cout << "New nbr list: ";
          for(uint64_t k=0; k < num_nbrs; k++) {
            std::cout << nbr_nbrs[k] << " ";
          }
          std::cout << std::endl;
          */
        }
        
        // upsert neighbor's neighbor list
        bool ret = this->Upsert(nbr_id, nbr_nbrs.data(), num_nbrs, nbr_gen);

        // retry if 'atomic' upsert failed
        retry = !ret;
        if (retry) {
          std::cout << "Upsert failed for node " << nbr_id << ", retrying" << std::endl;
        }
      }
    }
  
    return new_id;
  }

};
} // namespace diskann