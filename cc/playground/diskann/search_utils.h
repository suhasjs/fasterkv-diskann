// Utils to help with GreedySearch in Vamana
// Author: Suhas Jayaram Subramanya (suhasj@cs.cmu.edu)
#pragma once

#include <algorithm>
#include <cstdint>
#include <queue>
#include <unordered_set>

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
  FastCandidatePQ() : _size(0) {}

  // insert a candidate into the array, sort the array
  void push(const Candidate &c) {
    if (_size < PQ_DEFAULT_SIZE) {
      _array[_size++] = c;
      std::sort(_array, _array + _size, compare);
    } else {
      if (compare(c, _array[_size - 1])) {
        _array[_size - 1] = c;
        std::sort(_array, _array + _size, compare);
      }
    }
  }

  // pop the worst candidate, return a copy of the popped candidate
  Candidate pop_worst() {
    if (_size > 0) {
      return _array[--_size];
    }
    return Candidate();
  }

  // pop the best candidate, return a copy of the popped candidate
  Candidate pop_best() {
    if (_size > 0) {
      Candidate c = _array[0];
      _array[0] = _array[--_size];
      std::sort(_array, _array + _size, compare);
      return c;
    }
    return Candidate();
  }

  // return the current size of the candidate list
  uint32_t size() const { return _size; }

  // return a const reference to the best candidate
  const Candidate &best() const { return _array[0]; }

  // return a const reference to the worst candidate
  const Candidate &worst() const { return _array[_size - 1]; }

private:
  Candidate _array[PQ_DEFAULT_SIZE];
  uint32_t _size;
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

} // namespace diskann