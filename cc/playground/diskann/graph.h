// Implementation of graph data structure for DiskANN
// variable num_nbrs value implementation borrowed from test/compact_test.cc
// fixed size key implementation borrowed from test/test_types.h
// Author: Suhas Jayaram Subramanya (suhasj@cs.cmu.edu)

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

namespace diskann {
using namespace std::chrono_literals;
using namespace FASTER::core;
// fixed size key implementation borrowed from test/test_types.h
template <class T, class HashFn = std::hash<T>> class FixedSizeKey {
public:
  FixedSizeKey(T value) : key{value} {}

  FixedSizeKey(const FixedSizeKey &) = default;

  inline static constexpr uint32_t size() {
    return static_cast<uint32_t>(sizeof(FixedSizeKey));
  }

  inline FASTER::core::KeyHash GetHash() const {
    HashFn hash_fn;
    return FASTER::core::KeyHash{hash_fn(key)};
  }

  inline bool operator==(const FixedSizeKey &other) const {
    return key == other.key;
  }
  inline bool operator!=(const FixedSizeKey &other) const {
    return key != other.key;
  }

  T key;
};
using Key = FixedSizeKey<uint32_t>;

// forward declaration to declare friends for FlexibleValue
class GraphUpsertContext;
class GraphReadContext;
// variable num_nbrs value implementation borrowed from test/compact_test.cc
// size: size of buffer needed to hold this value object in memory (bytes)
// num_nbrs: number of neighbors contained in this FlexibleValue
class FlexibleValue {
public:
  FlexibleValue() : size_{sizeof(FlexibleValue)}, num_nbrs_{0} {}

  inline uint32_t size() const {
    return sizeof(*this) + num_nbrs_ * sizeof(uint32_t);
  }

  // set and get values for this value object
  friend class GraphUpsertContext;
  friend class GraphReadContext;

private:
  // size of this value object in bytes = sizeof(FlexibleValue) + num_nbrs_ *
  // sizeof(uint32_t)
  uint32_t size_;
  // sizeof(uint32_t) bytes per neighbor
  uint32_t num_nbrs_;

  // where to read the data in this FlexibleValue object?
  inline const uint32_t *buffer() const {
    return reinterpret_cast<const uint32_t *>(this + 1);
  }
  // where to write data into in this FlexibleValue object?
  inline uint32_t *buffer() { return reinterpret_cast<uint32_t *>(this + 1); }
};

// context to upsert a node + its neighbors into the graph
class GraphUpsertContext : public IAsyncContext {
public:
  typedef Key key_t;
  typedef FlexibleValue value_t;

  // key: node ID
  // nbrs: list of neighbors
  // num_nbrs: number of neighbors
  GraphUpsertContext(uint32_t key, uint32_t *nbrs, uint32_t num_nbrs)
      : key_{key}, nbrs_(nbrs), num_nbrs_{num_nbrs} {}

  /// Copy (and deep-copy) constructor.
  GraphUpsertContext(const GraphUpsertContext &other)
      : key_{other.key_}, nbrs_(other.nbrs_), num_nbrs_{other.num_nbrs_} {}

  /// The implicit and explicit interfaces require a key() accessor.
  inline const Key &key() const { return key_; }
  // size of this value object in bytes = sizeof(FlexibleValue) + 4B * num_nbrs_
  inline uint32_t value_size() const {
    return sizeof(FlexibleValue) + (num_nbrs_ * sizeof(uint32_t));
  }

  /// Non-atomic and atomic Put() methods.
  inline void Put(FlexibleValue &value) {
    // store num_nbrs and buffer in this value object
    value.num_nbrs_ = num_nbrs_;
    std::memcpy((void *)value.buffer(), (void *)nbrs_,
                num_nbrs_ * sizeof(uint32_t));
  }

  inline bool PutAtomic(FlexibleValue &value) {
    std::cout
        << "PutAtomic() called on node ID: {key_.key}, no atomic PUT available"
        << std::endl;
    // In-place update overwrites num_nbrs and buffer, but not size.
    value.num_nbrs_ = num_nbrs_;
    std::memcpy((void *)value.buffer(), (void *)nbrs_,
                num_nbrs_ * sizeof(uint32_t));
    return true;
  }

protected:
  /// The explicit interface requires a DeepCopy_Internal() implementation.
  Status DeepCopy_Internal(IAsyncContext *&context_copy) {
    return IAsyncContext::DeepCopy_Internal(*this, context_copy);
  }

private:
  Key key_;
  uint32_t *nbrs_;
  uint32_t num_nbrs_;
};

class GraphReadContext : public IAsyncContext {
public:
  typedef Key key_t;
  typedef FlexibleValue value_t;

  // key: node ID
  // output_buffer:
  GraphReadContext(uint32_t key, uint32_t *output_buffer,
                   uint32_t *output_num_nbrs)
      : key_{key}, output_nbrs{output_buffer}, output_num_nbrs{
                                                   output_num_nbrs} {}

  /// Copy (and deep-copy) constructor.
  GraphReadContext(const GraphReadContext &other)
      : key_{other.key_}, output_nbrs{other.output_nbrs},
        output_num_nbrs{other.output_num_nbrs} {}

  /// The implicit and explicit interfaces require a key() accessor.
  inline const Key &key() const { return key_; }

  inline void Get(const FlexibleValue &value) {
    // set number of nbrs
    *output_num_nbrs = value.num_nbrs_;
    // copy nbrs into output buffer
    std::memcpy((void *)output_nbrs, (void *)value.buffer(),
                value.num_nbrs_ * sizeof(uint32_t));
  }
  inline void GetAtomic(const FlexibleValue &value) {
    // set number of nbrs
    *output_num_nbrs = value.num_nbrs_;
    // copy nbrs into output buffer
    std::memcpy((void *)output_nbrs, (void *)value.buffer(),
                value.num_nbrs_ * sizeof(uint32_t));
    // std::cout << "GetAtomic() called on node ID: " << key_.key << ", no
    // atomic GET" << std::endl;
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
};

typedef FasterKv<Key, FlexibleValue, FASTER::device::NullDisk> MemGraph;

} // namespace diskann