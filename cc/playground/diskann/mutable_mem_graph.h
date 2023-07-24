// Implementation of mutable concurrent graph data structure for DiskANN
// fixed size key implementation borrowed from test/test_types.h
// variable size value implementation borrowed from test/in_memory_test.cc:451
// generation-lock for value implementation borrowed from test
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

// using diskann namespace to avoid namespace clashes with FASTER
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

// GenLock and AtomicGenLock borrowed from in_memory_test.cc:451
class GenLock {
public:
  GenLock() : control_{0} {}
  GenLock(uint64_t control) : control_{control} {}
  inline GenLock &operator=(const GenLock &other) {
    control_ = other.control_;
    return *this;
  }

  union {
    struct {
      uint64_t gen_number : 62;
      uint64_t locked : 1;
      uint64_t replaced : 1;
    };
    uint64_t control_;
  };
};
static_assert(sizeof(GenLock) == 8, "sizeof(GenLock) != 8");

class AtomicGenLock {
public:
  AtomicGenLock() : control_{0} {}
  AtomicGenLock(uint64_t control) : control_{control} {}

  inline GenLock load() const { return GenLock{control_.load()}; }
  inline void store(GenLock desired) { control_.store(desired.control_); }

  inline bool try_lock(bool &replaced) {
    replaced = false;
    GenLock expected{control_.load()};
    expected.locked = 0;
    expected.replaced = 0;
    GenLock desired{expected.control_};
    desired.locked = 1;

    if (control_.compare_exchange_strong(expected.control_,
                                          desired.control_)) {
      return true;
    }
    if (expected.replaced) {
      replaced = true;
    }
    return false;
  }
  inline void unlock(bool replaced) {
    if (!replaced) {
      // Just turn off "locked" bit and increase gen number.
      uint64_t sub_delta = ((uint64_t)1 << 62) - 1;
      control_.fetch_sub(sub_delta);
    } else {
      // Turn off "locked" bit, turn on "replaced" bit, and increase gen
      // number
      uint64_t add_delta = ((uint64_t)1 << 63) - ((uint64_t)1 << 62) + 1;
      control_.fetch_add(add_delta);
    }
  }

private:
  std::atomic<uint64_t> control_;
};
static_assert(sizeof(AtomicGenLock) == 8, "sizeof(AtomicGenLock) != 8");

// forward declaration to declare friends for MutableFlexibleValue
class MutableUpsertContext;
class MutableReadContext;
// mutable variable size value implementation borrowed from test/in_memory_test.cc:451
// size: size of buffer needed to hold this value object in memory (bytes)
// num_nbrs_: number of elements of size sizeof(T) contained in this object
// gen_lock_: generation lock for this value object
template <typename T> class MutableFlexibleValue {
public:
  MutableFlexibleValue() : gen_lock_{0}, size_{sizeof(MutableFlexibleValue<T>)}, num_nbrs_{0} {}

  // size of this object in bytes --> must be set by creator before submitting
  // to FASTER
  inline uint32_t size() const { return size_; }

  // set and get values for this value object
  friend class MutableUpsertContext;
  friend class MutableReadContext;

protected:
  // size of this value object in bytes = sizeof(MutableFlexibleValue) + num_nbrs_ *
  // sizeof(T)
  uint32_t size_;
  // sizeof(T) bytes per element
  uint32_t num_nbrs_;
  // generation lock for updates
  AtomicGenLock gen_lock_;

  // where to read the data in this MutableFlexibleValue object?
  inline const T *buffer() const {
    return reinterpret_cast<const T *>(this + 1);
  }
  // where to write data into in this MutableFlexibleValue object?
  inline T *buffer() { return reinterpret_cast<T *>(this + 1); }
};

using MutFlexValue = MutableFlexibleValue<uint32_t>;

// context to upsert a node + its neighbors into the graph
class MutableUpsertContext : public IAsyncContext {
public:
  typedef Key key_t;
  typedef MutFlexValue value_t;

  // key: node ID
  // nbrs: list of neighbors
  // num_nbrs: number of neighbors
  MutableUpsertContext(uint32_t key, uint32_t *nbrs, uint32_t num_nbrs, 
                       uint64_t expected_gen_number = std::numeric_limits<uint64_t>::max())
                     : key_{key}, nbrs_(nbrs), num_nbrs_{num_nbrs}, 
                       expected_gen_number_{expected_gen_number} {}

  /// Copy (and deep-copy) constructor.
  MutableUpsertContext(const MutableUpsertContext &other)
      : key_{other.key_}, nbrs_(other.nbrs_), 
        num_nbrs_{other.num_nbrs_}, expected_gen_number_{other.expected_gen_number_} {}

  /// The implicit and explicit interfaces require a key() accessor.
  inline const Key &key() const { return key_; }
  // size of this value object in bytes = sizeof(MutableFlexibleValue) + 4B * num_nbrs_
  inline uint32_t value_size() const {
    return sizeof(MutFlexValue) + (num_nbrs_ * sizeof(uint32_t));
  }

  /// Non-atomic and atomic Put() methods.
  inline void Put(MutFlexValue &value) {
    value.gen_lock_.store(0);
    value.size_ =
        sizeof(MutFlexValue) + (num_nbrs_ * sizeof(uint32_t));
    // store num_nbrs and buffer in this value object
    value.num_nbrs_ = num_nbrs_;
    std::memcpy((void *)value.buffer(), (void *)nbrs_,
                num_nbrs_ * sizeof(uint32_t));
  }

  inline bool PutAtomic(MutFlexValue &value) {
    // gen number will change if some other thread updated value between Read and PutAtomic
    if (expected_gen_number_ != std::numeric_limits<uint64_t>::max() &&
        expected_gen_number_ != value.gen_lock_.load().gen_number) {
      std::cout << "PutAtomic failed:: Expected gen #: " << expected_gen_number_ << ", actual gen #: " << value.gen_lock_.load().gen_number << std::endl;
      // return fail --> caller will retry (read -> inter-insert -> put)
      return false;
    }

    bool replaced;
    while (!value.gen_lock_.try_lock(replaced) && !replaced) {
      std::this_thread::yield();
    }
    if (replaced) {
      // Some other thread replaced this record.
      return false;
    }
    uint64_t new_size = sizeof(MutFlexValue) + (num_nbrs_ * sizeof(uint32_t));
    if (value.size_ < new_size) {
      // Current value is too small for in-place update.
      value.gen_lock_.unlock(true);
      return false;
    }
    // In-place update overwrites length and buffer, but not size.
    value.num_nbrs_ = num_nbrs_;
    std::memcpy((void *)value.buffer(), (void *)nbrs_,
                num_nbrs_ * sizeof(uint32_t))
    value.gen_lock_.unlock(false);
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
  uint64_t expected_gen_number_;
};

class MutableReadContext : public IAsyncContext {
public:
  typedef Key key_t;
  typedef MutFlexValue value_t;

  // key: node ID
  // output_buffer:
  MutableReadContext(uint32_t key, uint32_t *output_nbrs,
                   uint32_t *output_num_nbrs, uint64_t *output_gen_number = nullptr)
      : key_{key}, output_nbrs{output_nbrs}, output_num_nbrs{output_num_nbrs}, 
        output_gen_number{output_gen_number} {}

  /// Copy (and deep-copy) constructor.
  MutableReadContext(const MutableReadContext &other)
      : key_{other.key_}, output_nbrs{other.output_nbrs},
        output_num_nbrs{other.output_num_nbrs}, output_gen_number{other.output_gen_number} {}

  /// The implicit and explicit interfaces require a key() accessor.
  inline const Key &key() const { return key_; }

  inline void Get(const MutFlexValue &value) {
    // can allow for stale reads ?
    // set number of nbrs
    *output_num_nbrs = value.num_nbrs_;
    // copy nbrs into output buffer
    std::memcpy((void *)output_nbrs, (void *)value.buffer(),
                (*output_num_nbrs * sizeof(uint32_t)));
    // set generation number
    if (output_gen_number != nullptr) {
      *output_gen_number = value.gen_lock_.load().gen_number;
    }
  }
  inline void GetAtomic(const MutFlexValue &value) {
    GenLock before, after;
    do {
      // obtain pre-read generation number
      before = value.gen_lock_.load();
      // set number of nbrs
      *output_num_nbrs = value.num_nbrs_;
      // copy nbrs into output buffer
      std::memcpy((void *)output_nbrs, (void *)value.buffer(),
                  (*output_num_nbrs * sizeof(uint32_t)));
      // obtain post-read generation number
      after = value.gen_lock_.load();
    } while (before.gen_number != after.gen_number);
    // set generation number
    if (output_gen_number != nullptr) {
      *output_gen_number = after.gen_number;
    }
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
  uint64_t *output_gen_number = nullptr;
};


typedef FasterKv<Key, MutFlexValue, FASTER::device::NullDisk> MutableGraph;

} // namespace diskann
