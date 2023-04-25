// Implementation of graph data structure for DiskANN
// variable num_nbrs value implementation borrowed from test/compact_test.cc
// fixed size key implementation borrowed from test/test_types.h
// Author: Suhas Jayaram Subramanya (suhasj@cs.cmu.edu)

#pragma once
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>

#include "../src/core/faster.h"
#include "../src/device/null_disk.h"
#include "../test/test_types.h"
#include "mem_graph.h"

#define DISKANN_FASTER_SEGMENT_SIZE (1ULL << 30)

namespace diskann {
using namespace std::chrono_literals;
using namespace FASTER::core;
// fixed size key implementation borrowed from test/test_types.h
using DiskannKey = diskann::FixedSizeKey<uint32_t>;

// forward declaration to declare friends for DiskannValue
template <typename> class DiskannUpsertContext;
template <typename> class DiskannReadContext;

// inherit from FlexibleValue to store vector + neighbors for a vertex
// size: size of buffer to hold this object in memory (bytes)
// num_nbrs: number of neighbors contained in this object
// data, dim: vector data, dimensions
template <typename T> class DiskannValue : public FlexibleValue<uint32_t> {
public:
  // initialize FlexibleValue with 0 num nbrs
  DiskannValue() : FlexibleValue<uint32_t>(), num_dims_{0} {
    this->size_ = sizeof(*this);
  }

  inline uint32_t size() const {
    return sizeof(*this) + this->num_el_ * sizeof(uint32_t) +
           this->num_dims_ * sizeof(T);
  }

  // set and get values for this value object
  friend class DiskannUpsertContext<T>;
  friend class DiskannReadContext<T>;

protected:
  // size of this value object in bytes:
  //  sizeof(*this) + num_nbrs_ * sizeof(uint32_t) +
  //  num_dims_ * sizeof(T)
  // sizeof(T) bytes per dim in vector
  uint32_t num_dims_;

  // where to read the data in this object?
  // data is stored as follows:
  // 1. num_dims_ T values for vector data
  // 2. num_nbrs_ uint32_t values for neighbor IDs
  inline const uint32_t *buffer() const {
    return reinterpret_cast<const uint32_t *>(this + 1);
  }
  // where to write data into in this FlexibleValue object?
  inline uint32_t *buffer() { return reinterpret_cast<uint32_t *>(this + 1); }
};

// context to upsert a node + its neighbors into the graph
template <typename T> class DiskannUpsertContext : public IAsyncContext {
public:
  typedef Key key_t;
  typedef DiskannValue<T> value_t;
  // key: node ID
  // nbrs: list of neighbors
  // num_nbrs: number of neighbors
  DiskannUpsertContext(uint32_t key, T *data, uint32_t num_dims, uint32_t *nbrs,
                       uint32_t num_nbrs)
      : key_{key},
        nbrs_(nbrs), num_nbrs_{num_nbrs}, data_{data}, num_dims_{num_dims} {}

  /// Copy (and deep-copy) constructor.
  DiskannUpsertContext(const DiskannUpsertContext &other)
      : key_{other.key_}, nbrs_(other.nbrs_), num_nbrs_{other.num_nbrs_},
        data_{other.data_}, num_dims_{other.num_dims_} {}

  /// The implicit and explicit interfaces require a key() accessor.
  inline const Key &key() const { return key_; }
  // size of value in bytes = sizeof(value_t) + 4B * num_nbrs_ + num_dims *
  // sizeof(T)
  inline uint32_t value_size() const {
    return sizeof(DiskannValue<T>) + (num_dims_ * sizeof(T)) +
           (num_nbrs_ * sizeof(uint32_t));
  }

  /// Non-atomic and atomic Put() methods.
  inline void Put(DiskannValue<T> &value) {
    // get buffer to put
    void *buffer = (void *)value.buffer();

    // 1. store vector data
    std::copy(data_, data_ + num_dims_, (T *)buffer);
    value.num_dims_ = num_dims_;

    // 2. store neighbor data
    buffer = (void *)((T *)buffer + num_dims_);
    value.num_el_ = num_nbrs_;
    std::copy(nbrs_, nbrs_ + num_nbrs_, (uint32_t *)buffer);
  }

  inline bool PutAtomic(DiskannValue<T> &value) {
    std::cout
        << "PutAtomic() called on node ID: {key_.key}, no atomic PUT available"
        << std::endl;
    this->Put(value);
    return true;
  }

protected:
  /// The explicit interface requires a DeepCopy_Internal() implementation.
  Status DeepCopy_Internal(IAsyncContext *&context_copy) {
    return IAsyncContext::DeepCopy_Internal(*this, context_copy);
  }

private:
  Key key_;
  // store order: 1. vector data, 2. neighbor data
  T *data_;
  uint32_t num_dims_;
  uint32_t *nbrs_;
  uint32_t num_nbrs_;
};

template <typename T> class DiskannReadContext : public IAsyncContext {
public:
  typedef Key key_t;
  typedef DiskannValue<T> value_t;

  // key: node ID
  // output_buffer:
  DiskannReadContext(uint32_t key, T *output_data, uint32_t *output_num_dims,
                     uint32_t *output_buffer, uint32_t *output_num_nbrs)
      : key_{key}, output_nbrs{output_buffer}, output_num_nbrs{output_num_nbrs},
        output_data{output_data}, output_num_dims{output_num_dims} {}

  /// Copy (and deep-copy) constructor.
  DiskannReadContext(const DiskannReadContext &other)
      : key_{other.key_}, output_nbrs{other.output_nbrs},
        output_num_nbrs{other.output_num_nbrs}, output_data{other.output_data},
        output_num_dims{other.output_num_dims} {}

  /// The implicit and explicit interfaces require a key() accessor.
  inline const Key &key() const { return key_; }

  inline void Get(const DiskannValue<T> &value) {
    // 1. copy vector data
    void *buffer = (void *)value.buffer();
    *output_num_dims = value.num_dims_;
    std::copy((T *)buffer, (T *)buffer + value.num_dims_, this->output_data);

    // 2. copy neighbor data
    *output_num_nbrs = value.num_el_;
    buffer = (void *)((T *)buffer + value.num_dims_);
    std::copy((uint32_t *)buffer, (uint32_t *)buffer + value.num_el_,
              output_nbrs);
  }
  inline void GetAtomic(const DiskannValue<T> &value) {
    // std::cout << "GetAtomic() called on node ID: {key_.key}, no atomic GET
    // available" << std::endl;
    this->Get(value);
  }

protected:
  /// The explicit interface requires a DeepCopy_Internal() implementation.
  Status DeepCopy_Internal(IAsyncContext *&context_copy) {
    return IAsyncContext::DeepCopy_Internal(*this, context_copy);
  }

private:
  Key key_;

public:
  uint32_t *output_num_nbrs = nullptr;
  uint32_t *output_nbrs = nullptr;
  T *output_data = nullptr;
  uint32_t *output_num_dims = nullptr;
};
typedef FASTER::environment::QueueIoHandler handler_t;

typedef FasterKv<
    FixedSizeKey<uint32_t>, DiskannValue<float>,
    FASTER::device::FileSystemDisk<FASTER::environment::QueueIoHandler,
                                   DISKANN_FASTER_SEGMENT_SIZE>>
    DiskGraph;
} // namespace diskann