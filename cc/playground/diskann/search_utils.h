// Utils to help with GreedySearch in Vamana
// Author: Suhas Jayaram Subramanya (suhasj@cs.cmu.edu)
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <queue>
#include <unordered_set>
#include <utility>

#define PQ_DEFAULT_SIZE 256

#include "../src/core/faster.h"

namespace diskann {

// generic compare function to compare two vectors of type T
// returns squared L2 distance between a and b
template <typename T>
inline float compare(const T *a, const T *b, const uint32_t dim);

// specialization for float, using openmp pragma simd for auto-vectorization
// ensure that dim is a multiple of 8, `a` and `b` are 32-byte aligned
template <>
inline float compare<float>(const float *a, const float *b,
                            const uint32_t dim) {
  float dist = 0.0f;
#pragma omp simd reduction(+ : dist)
  for (uint32_t i = 0; i < dim; i++) {
    float diff = a[i] - b[i];
    dist += diff * diff;
  }
  return dist;
}

struct Candidate {
  uint32_t id = 0;
  float dist = 0;
  // equality operators (checks only id)
  bool operator==(const Candidate &c) const { return (id == c.id); }
  bool operator!=(const Candidate &c) const { return (id != c.id); }
  // comparison operators (checks only dist)
  bool operator<(const Candidate &c) const { return (dist < c.dist); }
  bool operator>(const Candidate &c) const { return (dist > c.dist); }
};

// compare function for Candidate struct
inline bool closer_is_better(const Candidate &a, const Candidate &b) {
  return a.dist < b.dist;
}
inline bool further_is_better(const Candidate &a, const Candidate &b) {
  return a.dist > b.dist;
}

// a fast candidate priority queue that uses a static array of size
// PQ_DEFAULT_SIZE; sort after every insert/delete since it is cheap
// for very small array; do not use heap operations (push_heap, pop_heap)
// templated by the compare function to use (e.g.// closer_is_better)
// NOTE:
//      * array[0] = best candidate (wins comparison against all other
//      candidates)
//      * array[size-1] = worst candidate (loses comparison against all other
//      candidates)
// supports: push -- inserts + sorts the array
//           pop_worst -- pops the worst candidate, returns a copy of the popped
//           candidate
//           pop_best -- pops the best candidate, returns a copy of
//           the popped candidate size -- returns the current size of the
//           candidate list best -- returns a const reference to the best
//           candidate worst -- returns a const reference to the worst candidate
template <bool (*compare)(const Candidate &, const Candidate &)>
class FastCandidatePQ {
public:
  FastCandidatePQ(uint32_t max_size) : _size(0), _max_size(max_size) {
    assert(max_size < PQ_DEFAULT_SIZE);
    _array = _array1;
    _use_default = true;
  }

  // step1: sort `c`
  // step2: truncated-merge `c` with _array
  // only keep the best `max_size` candidates
  void push_batch(Candidate *c, uint32_t n) {
    // sort incoming array
    std::sort(c, c + n, compare);
    Candidate *_dest_array = _flip_array();
    uint32_t i = 0, j = 0, k = 0;
    // merge `c` and _array
    while (i < n && j < _size && k < _max_size) {
      if (compare(c[i], _array[j])) {
        _dest_array[k++] = c[i++];
      } else {
        _dest_array[k++] = _array[j++];
      }
    }
    // copy remaining elements from `c`
    while (i < n && k < _max_size) {
      _dest_array[k++] = c[i++];
    }
    // copy remaining elements from _array
    while (j < _size && k < _max_size) {
      _dest_array[k++] = _array[j++];
    }
    _size = k;
    _array = _dest_array;
  }

  // trim the array to best n
  void trim(uint32_t n) {
    if (_size > n) {
      _size = n;
    }
  }

  // copy the best n candidates into the out_arr
  // remove them from the array
  void pop_best_n(uint32_t n) {
    if (_size < n) {
      n = _size;
    }
    Candidate *_dest_array = _flip_array();
    std::copy(_array + n, _array + n + _size, _dest_array);
    _size -= n;
    _array = _dest_array;
  }

  // const read ptr
  const Candidate *data() const { return _array; }

  // return the current size of the candidate list
  uint32_t size() const { return _size; }

  // return a const reference to the best candidate
  const Candidate &best() const { return _array[0]; }

  // return a const reference to the worst candidate
  const Candidate &worst() const { return _array[_size - 1]; }

private:
  Candidate *_flip_array() {
    if (_use_default) {
      _use_default = false;
      return _array2;
    } else {
      _use_default = true;
      return _array1;
    }
  }
  // 2 arrays to store the candidates
  Candidate *_array = nullptr;
  bool _use_default = true;
  Candidate _array1[PQ_DEFAULT_SIZE];
  Candidate _array2[PQ_DEFAULT_SIZE];
  uint32_t _size;
  uint32_t _max_size;
};

// specializations of FastCandidatePQ for closer/further comparators
using CloserPQ = FastCandidatePQ<closer_is_better>;
using FurtherPQ = FastCandidatePQ<further_is_better>;

float compute_recall(uint32_t *gt, uint32_t *result, uint32_t gt_NN,
                     uint32_t k_NN, uint32_t num_queries) {
  float recall = 0.0f;
  for (uint32_t i = 0; i < num_queries; i++) {
    uint32_t cur_count = 0;
    std::unordered_set<uint32_t> gt_set, result_set;
    uint32_t *gt_ptr = gt + i * gt_NN;
    uint32_t *result_ptr = result + i * k_NN;
    result_set.insert(result_ptr, result_ptr + k_NN);
    // check only the first k_NN in the ground truth
    for (uint32_t j = 0; j < k_NN; j++) {
      if (result_set.find(gt_ptr[j]) != result_set.end()) {
        cur_count++;
      }
    }
    recall += (float)cur_count / (float)k_NN;
  }
  return recall / (float)num_queries;
}

// stats per query
struct QueryStats {
  uint64_t total_us = 0; // total time to process query in micros
  uint64_t io_us = 0;    // total time spent in IO
  uint64_t cpu_us = 0;   // total time spent in CPU

  uint64_t n_ios = 0;     // total # of IOs issued
  uint64_t read_size = 0; // total # of bytes read
  uint64_t n_cmps = 0;    // # dist cmps
  uint64_t n_hops = 0;    // # search iters

  // operator overloads
  QueryStats &operator+=(const QueryStats &rhs) {
    total_us += rhs.total_us;
    io_us += rhs.io_us;
    cpu_us += rhs.cpu_us;
    n_ios += rhs.n_ios;
    read_size += rhs.read_size;
    n_cmps += rhs.n_cmps;
    n_hops += rhs.n_hops;
    return *this;
  }
  QueryStats operator+(const QueryStats &rhs) const {
    QueryStats tmp(*this);
    tmp += rhs;
    return tmp;
  }
  QueryStats &operator/=(const float &rhs) {
    total_us /= rhs;
    io_us /= rhs;
    cpu_us /= rhs;
    n_ios /= rhs;
    read_size /= rhs;
    n_cmps /= rhs;
    n_hops /= rhs;
    return *this;
  }
  QueryStats operator/(const float &rhs) const {
    QueryStats tmp(*this);
    tmp /= rhs;
    return tmp;
  }
  // overload = operator
  QueryStats &operator=(const QueryStats &rhs) {
    total_us = rhs.total_us;
    io_us = rhs.io_us;
    cpu_us = rhs.cpu_us;
    n_ios = rhs.n_ios;
    read_size = rhs.read_size;
    n_cmps = rhs.n_cmps;
    n_hops = rhs.n_hops;
    return *this;
  }
};
} // namespace diskann