// code borrowed from github.com/Microsoft/DiskANN
// Author: Suhas Jayaram Subramaya (suhasj@cs.cmu.edu)

#pragma once
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

// taken from
// https://github.com/Microsoft/BLAS-on-flash/blob/master/include/utils.h
// round up X to the nearest multiple of Y
#define ROUND_UP(X, Y)                                                         \
  ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

#define DIV_ROUND_UP(X, Y) (((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0))

// round down X to the nearest multiple of Y
#define ROUND_DOWN(X, Y) (((uint64_t)(X) / (Y)) * (Y))

namespace diskann {
// classes to read/write from/to disk in a cached manner
class cached_ifstream;
class cached_ofstream;

// reads bin metadata from a file, and sets npts and dim accordingly
inline void get_bin_metadata(const std::string &filename, uint32_t &npts,
                             uint32_t &dim, const uint64_t offset = 0) {
  int npts_i32, dim_i32;
  std::ifstream reader;
  reader.open(filename, std::ios::in | std::ios::binary);
  if (offset != 0) {
    reader.seekg(offset, std::ios::beg);
  }
  // read metadata for #pts and #dims
  reader.read((char *)&npts_i32, sizeof(int));
  reader.read((char *)&dim_i32, sizeof(int));
  npts = (unsigned)npts_i32;
  dim = (unsigned)dim_i32;

  // std::cout << "Binary metadata (" << filename << "): npts = " << npts << ",
  // dim = " << dim << std::endl;

  reader.close();
}

// reads graph metadata from a file
/* Vamana Graph HEADER */
// bytes 0-7  [8B]: size of file in bytes (size_t) [want]
// bytes 8-11 [4B]: max_observed_degree (uint32_t) [ignore]
// bytes 12-15[4B]: id of start point (uint32_t) [want]
// bytes 16-23[8B]: number of frozen points (uint64_t) [ignore]
// Vamana Graph DATA: start from offset=24
// for each point:
//    bytes 0-3  [4B]: num neighbors (uint32_t)
//    bytes 4-(num_neighbors*4 - 1): neighbors of point (vector<uint32_t>)
inline void get_graph_metadata(const std::string &filename, size_t &filesize,
                               uint32_t &start_id) {
  std::ifstream reader;
  reader.open(filename, std::ios::in | std::ios::binary);
  // read filesize
  reader.read((char *)&filesize, sizeof(size_t));
  // skip max_observed_degree
  reader.seekg(4, std::ios::cur);
  // read start_id
  reader.read((char *)&start_id, sizeof(uint32_t));
  // skip number of frozen points

  // std::cout << "Graph metadata (" << filename << "): filesize = " << filesize
  // << ", start_id = " << start_id << std::endl;

  reader.close();
}

// reads a binary file and populates the data array
// template argument T is the type of the data array
// args: data - pointer to the data array
//       filename - name of the binary file
//       npts - number of points in the file
//       dim - dimension of each point
//       aligned_dim - if non-zero, the data array is assumed to be aligned to
//       this dimension
template <typename T>
inline void populate_from_bin(T *data, const std::string &filename,
                              const uint32_t npts, const uint32_t dim,
                              const uint32_t aligned_dim = 0,
                              const uint64_t offset = 0) {
  assert(data != nullptr);
  int npts_i32, dim_i32;
  std::ifstream reader;
  // std::cout << "Reading file: " << filename << "..." << std::endl;
  reader.open(filename, std::ios::in | std::ios::binary);
  if (offset != 0) {
    reader.seekg(offset, std::ios::beg);
  }
  // read metadata for #pts and #dims
  reader.read((char *)&npts_i32, sizeof(int));
  reader.read((char *)&dim_i32, sizeof(int));
  // std::cout << "Binary metadata (" << filename << "): npts = " << npts_i32 <<
  // ", dim = " << dim_i32 << std::endl;
  assert(npts == (unsigned)npts_i32);
  assert(dim == (unsigned)dim_i32);

  const uint32_t read_dim = (aligned_dim == 0) ? dim : aligned_dim;
  // read the data
  if (read_dim == dim) {
    // if the data is aligned, read it in one go
    reader.read((char *)data, (uint64_t) npts * (uint64_t) dim * sizeof(T));
  } else {
    // if the data is not aligned, read it point by point
    for (uint32_t i = 0; i < npts; i++) {
      reader.read((char *)(data + (uint64_t) i * (uint64_t) read_dim), dim * sizeof(T));
    }
  }
  // std::cout << "Finished reading npts: " << npts << ", dim: " << dim << "
  // from file: " << filename << std::endl;
  reader.close();
}

// saves data to a binary file
// template argument T is the type of the data array
// args: filename - name of the binary file
//       data - pointer to the data array
//       npts - number of points in the file
//       dim - dimension of each point
template <typename T>
inline size_t save_bin(const std::string &filename, T *data, size_t npts,
                       size_t ndims) {
  std::ofstream writer;
  writer.open(filename, std::ios::out | std::ios::binary);
  std::cout << "Writing bin: " << filename << std::endl;
  int npts_i32 = (int)npts, ndims_i32 = (int)ndims;
  size_t bytes_to_write = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
  // write metadata for bin
  writer.write((char *)&npts_i32, sizeof(int));
  writer.write((char *)&ndims_i32, sizeof(int));
  std::cout << "bin metadata: #pts = " << npts << ", #dims = " << ndims
            << ", size = " << bytes_to_write << "B" << std::endl;
  // write data
  writer.write((char *)data, npts * ndims * sizeof(T));
  // flush cache
  writer.flush();
  writer.close();
  std::cout << "Finished writing bin." << std::endl;
  return bytes_to_write;
}

} // namespace diskann