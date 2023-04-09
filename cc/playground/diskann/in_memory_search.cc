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

  // stop session
  index.StopSession();
  std::cout << "Stopped session " << std::endl;

  std::cout << "Exiting" << std::endl;
  return 0;
}
