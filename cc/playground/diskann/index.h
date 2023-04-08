// Implementation of a Vamana index (purely in-memory, using a FASTER NullDisk)
// Author: Suhas Jayaram Subramanya (suhasj@cs.cmu.edu)

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>

#include "graph.h"
#include "io_utils.h"

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

  // FASTER store params
  uint64_t faster_max_keys_ = 0;
  uint64_t faster_memory_size_ = 0;
  std::string faster_graph_path_ = "";

public:
  FasterVamanaIndex(const uint64_t num_points, const uint64_t dim,
                    const std::string &index_load_path)
      : num_points_{num_points}, dim_{dim} {
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
    std::cout << "Setting FASTER store memory size to "
              << this->faster_memory_size_ / (1 << 20)
              << " MB, per key memory = " << per_key_memory << std::endl;

    /*** 3. Create FASTER store ****/
    this->graph_ =
        new diskann::MemGraph(this->faster_max_keys_, this->faster_memory_size_,
                              this->faster_graph_path_);
    std::cout << "Created FASTER store for VamanaIndex" << std::endl;

    /*** 4. Load graph data ****/
    std::ifstream graph_reader(index_load_path,
                               std::ios::in | std::ios::binary);
    // 24-byte header for Vamana graph
    graph_reader.seekg(24, std::ios::beg);
    // read and upsert graph data
    uint32_t node_id, num_nbrs;
    uint32_t nbrs[approx_degree];
    for (uint32_t i = 0; i < this->num_points_; i++) {
      node_id = i;
      // read num neighbors
      graph_reader.read(reinterpret_cast<char *>(&num_nbrs), sizeof(uint32_t));
      // read all neighbros
      graph_reader.read(reinterpret_cast<char *>(nbrs),
                        sizeof(uint32_t) * num_nbrs);
      // upsert into FASTER store
      this->Upsert(node_id, nbrs, num_nbrs);
    }
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
    }
  }

  /*** Read ***/
  // Read adjacency list for a node
  // args: node_id: node ID (uint32_t)
  //       nbrs: array of neighbors (uint32_t*)
  //       num_nbrs: number of neighbors (uint32_t)
  // Ensure `nbrs` has space to write least `max_degree_` elements
  void Read(uint32_t node_id, uint32_t *nbrs, uint32_t &num_nbrs) {
    std::cout << "Reading key:" << node_id << std::endl;
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

    std::cout << "Read " << num_nbrs << " neighbors for key " << node_id
              << std::endl;
  }
};
} // namespace diskann