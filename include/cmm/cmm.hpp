// github.com/cwpearson/cuda-memory-management

#pragma once

#include <cstdlib>
#include <new>

namespace cmm {

#define CMM_OR_THROW(ret, exc)                                                 \
  {                                                                            \
    if (ret != cudaSuccess)                                                    \
      throw exc;                                                               \
  }

namespace detail {

/* round x up to nearest multiple of `up`.
Up must not be 0
*/
inline size_t round_up(size_t x, size_t up) { return (x + up - 1) / up * up; }
} // namespace detail

template <class T> struct ZeroCopy {
private:
  bool set_;   // should set device
  int device_; // the device to set
public:
  typedef T value_type;
  ZeroCopy() : set_(false) {}
  explicit ZeroCopy(int dev) : set_(true), device_(dev) {}

  template <class U>
  ZeroCopy(const ZeroCopy<U> &other)
      : set_(other.set_), device_(other.device_) {}
  T *allocate(std::size_t n) {
    if (n > std::size_t(-1) / sizeof(T))
      throw std::bad_alloc();
    T *p = nullptr;
    if (set_) {
      CMM_OR_THROW(cudaSetDevice(device_), std::bad_alloc());
    }
    CMM_OR_THROW(cudaHostAlloc(&p, n * sizeof(T),
                               cudaHostAllocMapped | cudaHostAllocPortable),
                 std::bad_alloc());
    return p;
  }
  void deallocate(T *p, std::size_t) noexcept { cudaFreeHost(p); }
};
template <class T, class U>
bool operator==(const ZeroCopy<T> &, const ZeroCopy<U> &) {
  return true;
}
template <class T, class U>
bool operator!=(const ZeroCopy<T> &, const ZeroCopy<U> &) {
  return false;
}

/*
 */
template <class T> struct Malloc {
private:
  bool set_;   // should set device
  int device_; // the device to set
public:
  typedef T value_type;
  Malloc() : set_(false) {}
  explicit Malloc(int dev) : set_(true), device_(dev) {}
  template <class U>
  Malloc(const Malloc<U> &other) : set_(other.set_), device_(other.device_) {}
  T *allocate(std::size_t n) {
    if (n > std::size_t(-1) / sizeof(T))
      throw std::bad_alloc();
    T *p = nullptr;
    if (set_) {
      CMM_OR_THROW(cudaSetDevice(device_), std::bad_alloc());
    }
    CMM_OR_THROW(cudaMalloc(&p, n * sizeof(T)), std::bad_alloc());
    return p;
  }
  void deallocate(T *p, std::size_t) noexcept { cudaFree(p); }
};
template <class T, class U>
bool operator==(const Malloc<T> &, const Malloc<U> &) {
  return true;
}
template <class T, class U>
bool operator!=(const Malloc<T> &, const Malloc<U> &) {
  return false;
}

/*
 */
template <class T> class Managed {
private:
  bool set_;   // should set device
  int device_; // the device to set

public:
  typedef T value_type;
  Managed() : set_(false) {}
  explicit Managed(int dev) : set_(true), device_(dev) {}
  template <class U>
  Managed(const Managed<U> &other) : set_(other.set_), device_(other.device_) {}

  T *allocate(std::size_t n) {
    if (n > std::size_t(-1) / sizeof(T))
      throw std::bad_alloc();
    T *p = nullptr;
    if (set_) {
      CMM_OR_THROW(cudaSetDevice(device_), std::bad_alloc());
    }
    CMM_OR_THROW(cudaMallocManaged(&p, n * sizeof(T)), std::bad_alloc());
    return p;
  }
  void deallocate(T *p, std::size_t) noexcept { cudaFree(p); }
};
template <class T, class U>
bool operator==(const Managed<T> &, const Managed<U> &) {
  return true;
}
template <class T, class U>
bool operator!=(const Managed<T> &, const Managed<U> &) {
  return false;
}

#if 0
// https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/cuda-vmm/cuvector.h
template <class T> struct MemMap {
  typedef T value_type;

  CUdeviceptr d_p;
  CUmemAllocationProp prop;
  CUmemAccessDesc accessDesc;

  struct Range {
    CUdeviceptr start;
    size_t sz;
  };

  std::vector<CUmemGenericAllocationHandle> handles_;
  std::vector<Range> va_ranges;
  std::vector<size_t> handleSizes;

  size_t alloc_sz;
  size_t reserve_sz;
  size_t chunk_sz;

  MemMap() : ptr_(nullptr), size_(0) {}

  // Reserves some extra space in order to speed up grow()
  CUresult reserve(size_t new_sz);

  // Actually commits num bytes of additional memory
  CUresult grow(size_t new_sz);
};
#endif

} // namespace cmm

#undef CMM_OR_THROW