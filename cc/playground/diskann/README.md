# FASTER KV-backed DiskANN indices
This directory contains three prototype implementations --
- `in_memory_search`: loads a pre-built **`float`** Vamana index (purely in-memory) into a FASTER KV with a `NullDisk` store (purely in-memory) and runs multi-threaded search (1 query per thread)
- `pq_disk_search`: loads a pre-built **`float`** DiskANN index (PQ + disk-layout) into a FASTER KV with a `FilesystemDisk` store (minimum DRAM footprint, mostly on disk) and runs multi-threade search (1 query per thread)
- `in_memory_insert`: loads a pre-built **`float`** Vamana index (purely in-memory) into a FASTER KV with a `NullDisk` store (also purely in-memory), and inserts a set of points into the index (1 insert per thread)

Additional scripts --
- `scripts/partition_base_and_insert.py`: partitions a `source` file into two parts -- `base` and `insert`. Base contains points that will be indexed directly using `build_memory_index` from DiskANN repo, and `insert` contains points that will be inserted into the index using `in_memory_insert` from this repo. 
- **usage**: `python3 partition_base_and_insert.py --source_file <source_file> --base_file <base_file> --insert_file <insert_file> --num_base_pts <num_base_pts>`
- `base_file` will contain the first `num_base_pts` points from `source_file`, and `insert_file` will contain the remaining points from `source_file`
- *WARNING*: This script only works for vectors of type `float`/`uint32`/`int32` only (must be **exactly** 4 bytes per vector element)

## How to build?
*WARNING:* Build tested only on Ubuntu 20.04 and currently only supports **float** indices.
- Install build requirements -- `sudo apt install g++ libaio-dev uuid-dev libtbb-dev`
- Clone this repo to `fasterkv-diskann`, then `cd fasterkv-diskann/cc` and `mkdir build && cd build`
- Run Cmake (this will also clone `googleperftest`) -- `cmake -DCMAKE_BUILD_TYPE=Release ..`
- Build -- `make in_memory_search pq_disk_search in_memory_insert -j`


## How to run?
- In-memory search
  - **usage**: `./in_memory_search <index_prefix> <num_pts> <dim> <query_bin> <gt_bin> <k> <L> <beam_width> <num_threads>`
  - `index_prefix` - index path prefix for Vamana indices generated using DiskANN repo's `build_memory_index`
  - `num_pts` - num points in index
  - `dim` - num dimensions in each vector (must be `float`)
  - `query_bin` - path to query points in `fbin` format
  - `gt_bin` - path to ground-truth in `fbin` format
  - `k, L, beam_width, num_threads` - analogous to DiskANN repo's `search_memory_index` parameters
- Disk-based search
  - **usage**: `./pq_disk_search <index_prefix> <num_pts> <dim> <query_bin> <gt_bin> <k> <L> <beam_width> <num_threads>`
- In-memory insert
  - **usage**: `./in_memory_insert <index_prefix> <max_num_pts> <insert_file> <num_inserts> <insert_R> <insert_L_index> <insert_alpha> <insert_num_threads>`
  - `index_prefx` - index path prefix for Vamana indices generated using DiskANN repo's `build_memory_index`
  - `max_num_pts` - max num points in index (must be >= `num_pts` in `build_memory_index` + `num_inserts`)
  - `insert_file` - path to file containing points to insert in `fbin` format
  - `num_inserts` - num points to insert from `insert_file` into the KV-backed index
  - `insert_R` - max-degree for inserted points (also affects any points touched by the insertion during `inter_insert` for inserted points)
  - `insert_L_index` - candidate list size for inserted points
  - `insert_alpha` - alpha parameter for insertions
  - `insert_num_threads` - num threads to use for insertions (1 insert per thread)

## How to tune performance?
- FASTER KV stores data in chunks of a `page`. Further, FASTER requires **at-least** 6 pages of capacity for its in-memory log. See this [commit](https://github.com/suhasjs/fasterkv-diskann/commit/19fcbe04886d2aff2d3a1432c42a34371aedac0a#diff-fb1cd76bbbdd4821ac710a901d30d044094e58ebac755e049fce6ae229964f2f) on how to modify `src/core/address.h` to modify page size. You can use this to tune the amount of memory available for in-memory cache in the disk-based search
- This codebase makes heavy use of memory caching + re-use and assumes some graph parameters are within some chosen limits (max graph degree, num PQ chunks, max beam width) to pre-allocate memory for caching + re-use. Check the preprocessor defs in `faster_vamana.h`, `faster_diskann.h`, `pq_utils.h` to make sure input indices do not violate the chosen limits.

