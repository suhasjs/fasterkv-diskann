// Driver program to perform in-memory search on a FASTER-backed DiskANN index.
// Author: Suhas Jayaram Subramaya (suhasj@cs.cmu.edu)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../src/core/faster.h"
#include "faster_diskann.h"
#include <omp.h>

#define BLITZ_CACHE_SIZE (1 << 10)
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
  if (argc < 10) {
    std::cout << "Usage (only float vectors supported): " << argv[0]
              << " <index_prefix> <num_pts> <dim> <query_bin> <gt_bin> <k> <L> "
                 "<beam_width> <num_threads>"
              << std::endl;
    exit(0);
  }

  // parse args
  std::string index_prefix = argv[1];
  uint32_t num_pts = std::stoi(argv[2]);
  uint32_t dim = std::stoi(argv[3]);
  std::string query_bin = argv[4];
  std::string gt_bin = argv[5];
  uint32_t k_NN = std::stoi(argv[6]);
  uint32_t L_search = std::stoi(argv[7]);
  uint32_t beam_width = std::stoi(argv[8]);
  uint32_t num_threads = std::stoi(argv[9]);

  // create index
  diskann::FasterDiskANNIndex index(index_prefix);
  // start a session (akin to FASTER KV::StartSession)
  index.StartSession();

  // load index
  std::cout << "Loading index " << std::endl;
  index.load();
  index.load_pq();

  uint64_t max_degree = index.get_max_degree();

  // cache BFS levels 0-N
  // index.cache_bfs_levels(BLITZ_CACHE_SIZE);

  // load query data
  uint32_t num_queries, query_dim = dim;
  diskann::get_bin_metadata(query_bin, num_queries, query_dim);
  std::cout << "Loading query data from " << query_bin << " with "
            << num_queries << " queries of dimension " << query_dim
            << std::endl;
  uint32_t aligned_dim = ROUND_UP(query_dim, 16);
  uint64_t alloc_size = num_queries * aligned_dim * sizeof(float);
  // round up to nearest multiple of alignment val
  alloc_size = ROUND_UP(alloc_size, 1024);
  float *query_data =
      reinterpret_cast<float *>(FASTER::core::aligned_alloc(1024, alloc_size));
  diskann::populate_from_bin<float>(query_data, query_bin, num_queries,
                                    query_dim, aligned_dim);
  // load ground truth data
  uint32_t gk_kNN, num_gt = num_queries;
  diskann::get_bin_metadata(gt_bin, num_gt, gk_kNN);
  std::cout << "Loading ground truth data from " << gt_bin << " with " << num_gt
            << " ground truth lists of size " << gk_kNN << std::endl;
  uint64_t gt_alloc_size = num_gt * gk_kNN * sizeof(uint32_t);
  uint32_t *gt_data = new uint32_t[gt_alloc_size];
  diskann::populate_from_bin<uint32_t>(gt_data, gt_bin, num_gt, gk_kNN, gk_kNN);

  // create output bufer
  uint32_t *result = new uint32_t[k_NN * num_queries];
  float *result_dist = new float[k_NN * num_queries];
  diskann::QueryStats *result_stats = new diskann::QueryStats[num_queries];

  // setup query contexts
  std::vector<diskann::QueryContext *> query_contexts;
  for (uint32_t i = 0; i < num_threads; i++) {
    query_contexts.emplace_back(
        new diskann::QueryContext(beam_width, max_degree, L_search));
  }

  // times `func` on `args` and returns the time in microseconds
  // credit: https://stackoverflow.com/a/53498501
  auto time_query_us = [&index](auto &&... params) {
    // get time before function invocation
    const auto &start = std::chrono::high_resolution_clock::now();
    // function invocation using perfect forwarding
    index.search(std::forward<decltype(params)>(params)...);
    // get time after function invocation
    const auto &stop = std::chrono::high_resolution_clock::now();
    // return time difference in microseconds
    return std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
        .count();
  };

  // run search (sequential)
  const auto start_t = std::chrono::high_resolution_clock::now();
  num_queries = 1;
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
  for (uint32_t i = 0; i < num_queries; i++) {
    uint32_t thread_num = omp_get_thread_num();
    uint64_t start_tsc = __builtin_ia32_rdtsc();
    query_contexts[thread_num]->reset();
    // inputs to search
    float *cur_query = query_data + i * aligned_dim;
    uint32_t *cur_result = result + i * k_NN;
    float *cur_result_dist = result_dist + i * k_NN;
    // perform search
    uint64_t cur_query_time =
        time_query_us(cur_query, k_NN, L_search, cur_result, cur_result_dist,
                      result_stats + i, beam_width, query_contexts[thread_num]);
    uint64_t end_tsc = __builtin_ia32_rdtsc();
    result_stats[i].cpu_ticks = end_tsc - start_tsc;
    result_stats[i].total_us = cur_query_time;
  }
  const auto stop_t = std::chrono::high_resolution_clock::now();
  double time_seconds =
      std::chrono::duration_cast<std::chrono::microseconds>(stop_t - start_t)
          .count() *
      1.0 / 1e6;
  double qps = num_queries / time_seconds;

  // compute recall
  float recall =
      diskann::compute_recall(gt_data, result, gk_kNN, k_NN, num_queries);

  // compute avg stats
  diskann::QueryStats avg_stats = result_stats[0];
  avg_stats =
      std::accumulate(result_stats + 1, result_stats + num_queries, avg_stats);
  auto avg = [&num_queries](uint64_t val) { return val / (float)num_queries; };
  std::cout
      << "threads, k, L, beamwidth, qps, recall, latency, p99_latecy, cmps, "
         "hops, ios, iobytes, iosize, ioticks, cputicks"
      << std::endl;
  std::cout << num_threads << ", " << k_NN << ", " << L_search << ", "
            << beam_width << ", " << qps << ", " << recall * 100 << ", "
            << avg(avg_stats.total_us) << ", "
            << get_p99_latency(result_stats, num_queries) << ", "
            << avg(avg_stats.n_cmps) << ", " << avg(avg_stats.n_hops) << ", "
            << avg(avg_stats.n_ios) << ", " << avg(avg_stats.read_size) << ", "
            << avg_stats.read_size / avg_stats.n_ios << ", "
            << avg(avg_stats.io_ticks) << ", "
            << (uint64_t)avg(avg_stats.cpu_ticks) << std::endl;

  // stop session
  index.StopSession();

  // free any alloc'ed resources
  free(query_data);
  delete[] gt_data;
  delete[] result;
  delete[] result_dist;
  delete[] result_stats;
  for (auto &ctx : query_contexts) {
    delete ctx;
  }

  return 0;
}
