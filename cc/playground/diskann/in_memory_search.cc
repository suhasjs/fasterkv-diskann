#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "graph.h"

#include "../src/core/faster.h"

using namespace FASTER::core;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <graph> <query>" << std::endl;
    exit(0);
  }

  // instantiate a store with 2^10 keys, 1GB of memory
  diskann::MemIndex store{1024, 1 * 1024 * 1024 * 1024l, ""};

  std::cout << "Created store " << std::endl;

  // start a session
  store.StartSession();
  std::cout << "Started session " << argv[1] << std::endl;

  // Insert 256 keys
  uint32_t num_keys = 256;
  // create a vector of 64 uint32_t values
  for (size_t idx = 0; idx < num_keys; ++idx) {
    std::cout << "Inserting key:" << idx << std::endl;
    auto callback = [](IAsyncContext *ctxt, Status result) {
      std::cout
          << "Upsert completed for key: "
          << reinterpret_cast<diskann::GraphUpsertContext *>(ctxt)->key().key
          << std::endl;
    };
    std::vector<uint32_t> values(idx + 1, idx);
    uint32_t num_nbrs = values.size();
    diskann::GraphUpsertContext context{
        static_cast<uint32_t>(idx), reinterpret_cast<uint32_t *>(values.data()),
        static_cast<uint32_t>(values.size())};
    auto result = store.Upsert(context, callback, 1);
    if (result != Status::Ok) {
      std::cout << "Upsert failed with status " << result << std::endl;
    }
  }

  // stop session
  store.StopSession();

  // start a new session
  store.StartSession();

  // read all the inserted keys
  for (size_t idx = 0; idx < num_keys; ++idx) {
    std::cout << "Reading key:" << idx << std::endl;
    auto callback = [](IAsyncContext *ctxt, Status result) {
      std::cout
          << "Read completed for key: "
          << reinterpret_cast<diskann::GraphReadContext *>(ctxt)->key().key
          << std::endl;
    };
    // where to read into?
    std::vector<uint32_t> nbrs(64, 5000);
    uint32_t num_nbrs = 50000;
    diskann::GraphReadContext context{static_cast<uint32_t>(idx),
                                      reinterpret_cast<uint32_t *>(nbrs.data()),
                                      static_cast<uint32_t *>(&num_nbrs)};

    auto result = store.Read(context, callback, 1);
    if (result != Status::Ok) {
      std::cout << "Read failed with status " << result << std::endl;
    }

    std::cout << "Read " << num_nbrs << " neighbors for key " << idx
              << std::endl;

    // check if the read values are correct
    for (size_t i = 0; i < num_nbrs; ++i) {
      if (nbrs[i] != idx) {
        std::cout << "Read incorrect value " << nbrs[i] << " for key " << idx
                  << std::endl;
      }
    }
  }

  std::cout << "Exiting" << std::endl;
  return 0;
}
