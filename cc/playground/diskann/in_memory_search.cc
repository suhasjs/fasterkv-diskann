// Driver program to perform in-memory search on a FASTER-backed DiskANN index.
// Author: Suhas Jayaram Subramaya (suhasj@cs.cmu.edu)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../src/core/faster.h"
#include "index.h"

using namespace FASTER::core;

int main(int argc, char *argv[]) {
  // assume float data type for now
  if (argc < 8) {
    std::cout << "Usage (only float vectors supported): " << argv[0]
              << " <index_prefix> <num_pts> <dim> <query_bin> <gt_bin> <k> <L>"
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

  // create index
  std::cout << "Creating FasterVamanaIndex with FASTER NullDisk " << std::endl;
  diskann::FasterVamanaIndex index(num_pts, dim, index_prefix);
  // start a session (akin to FASTER KV::StartSession)
  index.StartSession();
  std::cout << "Started session " << std::endl;

  // load index
  std::cout << "Loading index " << std::endl;
  index.load();

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

  // run search (sequential)
  for (uint32_t i = 0; i < num_queries; i++) {
    // inputs to search
    float *cur_query = query_data + i * aligned_dim;
    uint32_t *cur_result = result + i * k_NN;
    float *cur_result_dist = result_dist + i * k_NN;
    // perform search
    index.search(cur_query, k_NN, L_search, cur_result, cur_result_dist);
  }

  // compute recall
  float recall =
      diskann::compute_recall(gt_data, result, gk_kNN, k_NN, num_queries);
  std::cout << "Recall: " << recall << std::endl;

  // stop session
  index.StopSession();
  std::cout << "Stopped session " << std::endl;

  // free any alloc'ed resources
  free(query_data);
  delete[] gt_data;
  delete[] result;
  delete[] result_dist;

  std::cout << "Exiting" << std::endl;
  return 0;
}
