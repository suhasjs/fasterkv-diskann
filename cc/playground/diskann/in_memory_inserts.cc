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
  if (argc < 9) {
    std::cout << "Usage (only float vectors supported): " << argv[0]
              << " <[1]index_prefix> <[2]max_num_pts> <[3]insert_file>"
              << " <[4]num_inserts> <[5]insert_R> <[6]insert_L_index> <[7]insert_alpha> <[8]insert_num_threads>"
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
  uint32_t num_threads = std::stoi(argv[8]);

  // create index
  diskann::FasterFreshVamanaIndex index(max_num_pts, max_degree, index_prefix);
  
  // start a session (akin to FASTER KV::StartSession)
  index.StartSession();

  // load index
  std::cout << "Loading index " << std::endl;
  index.load();

  // extra allocs for API reasons
  const uint32_t num_queries = 16, beam_width = 4, L_search = insert_L_index;
  diskann::QueryStats *result_stats = new diskann::QueryStats[num_queries];
  std::vector<diskann::QueryContext *> query_contexts;
  for (uint32_t i = 0; i < num_threads; i++) {
    query_contexts.emplace_back(
        new diskann::QueryContext(beam_width, max_degree, L_search));
  }

  // load insert vectors
  std::cout << "Loading insert vectors from " << insert_vec_file << std::endl;
  uint32_t num_insertfile_vecs, insert_dim;
  diskann::get_bin_metadata(insert_vec_file, num_insertfile_vecs, insert_dim);
  const uint32_t aligned_dim = ROUND_UP(insert_dim, 16);
  uint64_t insert_alloc_size = ((uint64_t) num_insertfile_vecs * (uint64_t) aligned_dim * sizeof(float));
  insert_alloc_size = ROUND_UP(insert_alloc_size, 1024);
  std::cout << "Allocating " << insert_alloc_size*1.0 / 1048576 << " MB for insert vectors." << std::endl;
  float* insert_vecs = reinterpret_cast<float *>(FASTER::core::aligned_alloc(1024, insert_alloc_size));
  // zero out the data array (to set unused dimensions to 0)
  memset(insert_vecs, 0, insert_alloc_size);
  // populate vector data from data file
  diskann::populate_from_bin<float>(insert_vecs, insert_vec_file, num_insertfile_vecs, insert_dim, aligned_dim);

  // begin insertion into index
  std::cout << "Inserting " << num_inserts << " vectors using " << num_threads << " threads." << std::endl;
  const auto &insert_start_t = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
  for(uint64_t i=0; i < num_inserts ;i++) {
    uint64_t thread_num = omp_get_thread_num();
    float *insert_vec = insert_vecs + (i * aligned_dim);
    diskann::QueryContext *ctx = query_contexts[thread_num];

    uint64_t inserted_id = index.insert(insert_vec, insert_L_index, insert_alpha, result_stats, ctx);
    // std::cout << "Vector #" << i << " ==> id " << inserted_id << std::endl;
  }
  const auto &insert_end_t = std::chrono::high_resolution_clock::now();
  uint64_t insert_duration = std::chrono::duration_cast<std::chrono::microseconds>(insert_end_t - insert_start_t).count();
  double insert_throughput = (double) num_inserts / ((double) insert_duration / 1000000.0);
  double insert_duration_s = (double) insert_duration / 1000000.0;
  double insert_throughput_per_thread = insert_throughput / (double) num_threads;
  std::cout << "Inserted " << num_inserts << " vectors in " << insert_duration_s << " seconds.\nThroughput: " << insert_throughput << " inserts/sec, " << insert_throughput_per_thread << " inserts/sec/thread" << std::endl;

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
