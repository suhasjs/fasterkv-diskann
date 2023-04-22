// Utils to help with GreedySearch in Vamana
// Author: Suhas Jayaram Subramanya (suhasj@cs.cmu.edu)
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <immintrin.h>
#include <queue>
#include <unordered_set>
#include <utility>

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
#pragma omp simd reduction(+ : dist) aligned(a, b : 32)
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
// _pq_alloc_size; sort after every insert/delete since it is cheap
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
  static FastCandidatePQ *create_from_array(uint8_t *obj_buf, uint32_t max_size,
                                            uint32_t alloc_size) {
    // allocate memory for the object
    FastCandidatePQ *obj = new (obj_buf) FastCandidatePQ();
    obj->_size = 0;
    obj->_max_size = max_size;
    obj->_pq_alloc_size = alloc_size;
    obj->_array = obj->_array1;
    obj->_use_default = true;

    // setup arrays
    Candidate *array1, *array2;
    void *next_ptr = (void *)(obj + 1);
    next_ptr = (void *)ROUND_UP((uint64_t)next_ptr, 16);
    array1 = (Candidate *)next_ptr;
    array2 = array1 + alloc_size;
    obj->_array1 = array1;
    obj->_array2 = array2;
    // test write to bounds of array
    obj->_array1[alloc_size - 1].dist = 0;
    obj->_array2[alloc_size - 1].dist = 0;
    return obj;
  }

  // step1: sort `c`
  // step2: truncated-merge `c` with _array
  // only keep the best `max_size` candidates
  void push_batch(Candidate *c, uint32_t n) {
    // sort incoming array
    std::sort(c, c + n, compare);
    Candidate *_dest_array = _flip_array();
    // prefetch the destination array
    _mm_prefetch((const char *)_dest_array, _MM_HINT_T0);
    _mm_prefetch((const char *)_array, _MM_HINT_T0);
    _mm_prefetch((const char *)c, _MM_HINT_T0);
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
  Candidate *_array1 = nullptr;
  Candidate *_array2 = nullptr;
  // cur size of PQ
  uint32_t _size = 0;
  // max allowed size of PQ
  uint32_t _max_size = 0;
  // backing array size for each _array
  uint32_t _pq_alloc_size = 0;
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
  uint64_t total_us = 0;  // total time to process query in micros
  uint64_t io_us = 0;     // time spent on IO in micros
  uint64_t cpu_us = 0;    // time spent on CPU in micros
  uint64_t io_ticks = 0;  // number of CPU ticks spent on IO
  uint64_t cpu_ticks = 0; // total number of CPU ticks

  uint64_t n_ios = 0;     // total # of IOs issued
  uint64_t read_size = 0; // total # of bytes read
  uint64_t n_cmps = 0;    // # dist cmps
  uint64_t n_hops = 0;    // # search iters

  // operator overloads
  QueryStats &operator+=(const QueryStats &rhs) {
    total_us += rhs.total_us;
    io_us += rhs.io_us;
    cpu_us += rhs.cpu_us;
    io_ticks += rhs.io_ticks;
    cpu_ticks += rhs.cpu_ticks;
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
    io_ticks /= rhs;
    cpu_ticks /= rhs;
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
    io_ticks = rhs.io_ticks;
    cpu_ticks = rhs.cpu_ticks;
    n_ios = rhs.n_ios;
    read_size = rhs.read_size;
    n_cmps = rhs.n_cmps;
    n_hops = rhs.n_hops;
    return *this;
  }
};

struct QueryContext {
  Candidate *cur_beam = nullptr;
  uint32_t cur_beam_size = 0;
  Candidate *beam_new_cands = nullptr;
  uint32_t beam_num_new_cands = 0;
  uint32_t *beam_nbrs_cache = nullptr;
  uint32_t **beam_nbrs = nullptr;
  uint32_t *beam_nnbrs = nullptr;
  uint8_t *buf =
      nullptr; // uint8_t type to make it easier to do pointer arithmetic
  uint64_t buf_size = 0;
  CloserPQ *unexplored_front = nullptr, *explored_front = nullptr;

  QueryContext(uint32_t beam_width, uint32_t max_degree, uint32_t L_search) {
    // compute buf size
    buf_size = 0;

    // round up to nearest multiple of 4 and 8 for alignment
    beam_width = ROUND_UP(beam_width, 8);
    max_degree = ROUND_UP(max_degree, 8);
    uint32_t alloc_L_search = ROUND_UP(1.5 * L_search, 8);
    uint64_t pq_alloc_size =
        ROUND_UP(2 * alloc_L_search * sizeof(Candidate), 16);
    uint64_t front_alloc_size =
        ROUND_UP(pq_alloc_size + 2 * sizeof(CloserPQ), 16);
    buf_size += beam_width * sizeof(Candidate);                // cur_beam
    buf_size += (beam_width * max_degree) * sizeof(Candidate); // beam_new_cands
    buf_size += (beam_width * max_degree) * sizeof(uint32_t); // beam_nbrs_cache
    buf_size += beam_width * sizeof(uint32_t *);              // beam_nbrs
    buf_size += beam_width * sizeof(uint32_t);                // beam_nnbrs
    buf_size += front_alloc_size; // unexplored_front, 2x arrays internally
    buf_size += front_alloc_size; // explored_front, 2x arrays internally

    // round up: 256 byte alignment
    buf_size = ROUND_UP(buf_size, 256);

    // alloc aligned buffer to buf
    this->buf = (uint8_t *)FASTER::core::aligned_alloc(256, buf_size);
    // std::cout << "Allocating QueryContext: alloc " << buf_size << " B" <<
    // std::endl;

    // set pointers within buf
    uint64_t offset = 0;
    // allocate cur_beam
    cur_beam = (Candidate *)(buf + offset);
    offset += beam_width * sizeof(Candidate);
    // allocate beam_new_cands
    beam_new_cands = (Candidate *)(buf + offset);
    offset += (beam_width * max_degree) * sizeof(Candidate);
    // allocate beam_nbrs_cache
    beam_nbrs_cache = (uint32_t *)(buf + offset);
    offset += (beam_width * max_degree) * sizeof(uint32_t);
    // allocate beam_nbrs
    beam_nbrs = (uint32_t **)(buf + offset);
    offset += beam_width * sizeof(uint32_t *);
    // allocate beam_nnbrs
    beam_nnbrs = (uint32_t *)(buf + offset);
    offset += beam_width * sizeof(uint32_t);
    // allocate unexplored_front arrays
    unexplored_front =
        CloserPQ::create_from_array(buf + offset, L_search, alloc_L_search);
    offset += front_alloc_size;
    // allocate explored_front arrays
    explored_front =
        CloserPQ::create_from_array(buf + offset, L_search, alloc_L_search);
    offset += front_alloc_size;
    // std::cout << "QueryContext: offset=" << offset << std::endl;
    // sanity check
    assert(ROUND_UP(offset, 256) == buf_size);

    // set beam_nbrs
    beam_nbrs[0] = beam_nbrs_cache;
    for (uint32_t i = 0; i < beam_width; i++) {
      beam_nbrs[i] = beam_nbrs_cache + i * max_degree;
    }
  }

  void reset() {
    cur_beam_size = 0;
    beam_num_new_cands = 0;
    unexplored_front->trim(0);
    explored_front->trim(0);
  }

  ~QueryContext() {
    if (buf != nullptr) {
      // std::cout << "Freeing QueryContext: released " << buf_size << " B" <<
      // std::endl;
      FASTER::core::aligned_free(buf);
    }
  }
};

} // namespace diskann