# FASTER KV-backed DiskANN indices
This directory contains two prototype implementations --
- `in_memory_search`: loads a pre-built **`float`** Vamana index (purely in-memory) into a FASTER KV with a `NullDisk` store (purely in-memory) and runs multi-threaded search (1 thread per query)
- `pq_disk_search`: loads a pre-built **`float`** DiskANN index (PQ + disk-layout) into a FASTER KV with a `FilesystemDisk` store (minimum DRAM footprint, mostly on disk) and runs multi-threade search (1 thread per query)

## How to build?
*WARNING:* Build tested only on Ubuntu 20.04 and currently only supports **float** indices.
- Install build requirements -- `sudo apt install g++ libaio-dev uuid-dev libtbb-dev`
- Clone this repo to `fasterkv-diskann`, then `cd fasterkv-diskann/cc` and `mkdir build && cd build`
- Run Cmake (this will also clone `googleperftest`) -- `cmake -DCMAKE_BUILD_TYPE=Release ..`
- Build -- `make in_memory_search pq_disk_search -j`


# How to run?
- In-memory search
  - `./in_memory_search <index_prefix> <num_pts> <dim> <query_bin> <gt_bin> <k> <L> <beam_width> <num_threads>`
  - `index_prefix` - index path prefix for Vamana indices generated using DiskANN repo's `build_memory_index` (e.g. )
  - `num_pts` - num points in index
  - `dim` - num dimensions in each vector (must be `float`)
  - `query_bin` - path to query points in `bin` format
  - `gt_bin` - path to ground-truth in `bin` format
  - `k, L, beam_width, num_threads` - analogous to DiskANN repo's `search_memory_index` parameters
- Disk-based search
  - `./pq_disk_search <index_prefix> <num_pts> <dim> <query_bin> <gt_bin> <k> <L> <beam_width> <num_threads>`


# How to tune performance?
- FASTER KV stores data in chunks of a `page`. Further, FASTER requires **at-least** 6 pages of capacity for its in-memory log. See this [commit](https://github.com/suhasjs/fasterkv-diskann/commit/19fcbe04886d2aff2d3a1432c42a34371aedac0a#diff-fb1cd76bbbdd4821ac710a901d30d044094e58ebac755e049fce6ae229964f2f) on how to modify `src/core/address.h` to modify page size. You can use this to tune the amount of memory available for in-memory cache in the disk-based search
- This codebase makes heavy use of memory caching + re-use and assumes some graph parameters are within some chosen limits (max graph degree, num PQ chunks, max beam width) to pre-allocate memory for caching + re-use. Check the preprocessor defs in `faster_vamana.h`, `faster_diskann.h`, `pq_utils.h` to make sure input indices do not violate the chosen limits.

