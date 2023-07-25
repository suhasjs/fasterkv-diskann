// Driver program that does the following:
// 1. Loads a Vamana index into NullDisk-backed FASTER KV store.
// 2. Inserts a new set of vectors into the FasterFreshVamana index.
// Intended to measure only the throughput of insert operations on the FasterFreshVamana index
// Author: Suhas Jayaram Subramaya (suhasj@cs.cmu.edu)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../src/core/faster.h"
#include "faster_fresh_vamana.h"
#include "search_utils.h"
#include "io_utils.h"
#include <omp.h>

using namespace FASTER::core;

uint64_t get_p99_latency(diskann::QueryStats *stats, uint32_t num_queries) {
  std::vector<uint64_t> latencies(num_queries);
  for (uint32_t i = 0; i < num_queries; i++) {
    latencies[i] = stats[i].total_us;
  }
  std::sort(latencies.begin(), latencies.end());
  uint64_t p99_idx = (uint64_t)(num_queries * 0.99);
  return latencies[p99_idx];
}

int main(int argc, char *argv[]) {
  // assume float data type for now
  if (argc < 8) {
    std::cout << "Usage (only float vectors supported): " << argv[0]
              << " <[1]index_prefix> <[2]max_num_pts> <[3]insert_file>"
              << " <[4]num_inserts> <[5]insert_R> <[6]insert_L_index> <[7]insert_alpha>"
              << std::endl;
    exit(0);
  }

  // parse args
  std::string index_prefix = argv[1];
  uint32_t max_num_pts = std::stoi(argv[2]);
  std::string insert_vec_file = argv[3];
  uint32_t num_inserts = std::stoi(argv[4]);
  uint32_t max_degree = std::stoi(argv[5]);
  uint32_t insert_L_index = std::stoi(argv[6]);
  float insert_alpha = std::stof(argv[7]);

  // create index
  diskann::FasterFreshVamanaIndex index(max_num_pts, max_degree, index_prefix);
  
  // start a session (akin to FASTER KV::StartSession)
  index.StartSession();

  // load index
  std::cout << "Loading index " << std::endl;
  index.load();

  // extra allocs for API reasons
  const uint32_t num_queries = 16, beam_width = 4, L_search = 50;
  const uint32_t num_threads = 16;
  diskann::QueryStats *result_stats = new diskann::QueryStats[num_queries];
  std::vector<diskann::QueryContext *> query_contexts;
  for (uint32_t i = 0; i < num_threads; i++) {
    query_contexts.emplace_back(
        new diskann::QueryContext(beam_width, max_degree, L_search));
  }

  // load insert vectors
  std::cout << "Loading insert vectors from " << insert_vec_file << std::endl;
  uint32_t num_insert, insert_dim;
  diskann::get_bin_metadata(insert_vec_file, num_insert, insert_dim);
  const uint32_t aligned_dim = ROUND_UP(insert_dim, 16);
  const uint64_t insert_alloc_size = ((uint64_t) num_insert * (uint64_t) aligned_dim * sizeof(float));
  float* insert_vecs = reinterpret_cast<float *>(FASTER::core::aligned_alloc(1024, insert_alloc_size));
  // zero out the data array (to set unused dimensions to 0)
  memset(insert_vecs, 0, insert_alloc_size);
  // populate vector data from data file
  diskann::populate_from_bin<float>(insert_vecs, insert_vec_file, num_insert, insert_dim, aligned_dim);

  for(uint64_t i=0; i < 1;i++) {
    std::cout << "index.insert() --> #" << i << std::endl;
    uint64_t inserted_id = index.insert(insert_vecs + (i * aligned_dim), insert_L_index, insert_alpha, result_stats, query_contexts[0]);
    std::cout << "Vector inserted into index with id " << inserted_id << std::endl;
  }

  // stop session
  index.StopSession();

  // free any alloc'ed resources
  FASTER::core::aligned_free(insert_vecs);
  delete[] result_stats;
  for (auto &ctx : query_contexts) {
    delete ctx;
  }

  return 0;
}
