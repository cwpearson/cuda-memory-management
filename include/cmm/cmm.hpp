// github.com/cwpearson/cuda-memory-management

#pragma once

#include <cstdlib>
#include <new>

#include <cuda.h>

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
inline size_t round_up(size_t x, size_t up) { return ((x + up - 1) / up) * up; }
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

#if 1
// https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/cuda-vmm/cuvector.h
template <class T> struct Buffer {
  typedef T value_type;

  // VA pointer to memory
  CUdeviceptr d_p;
  CUmemAllocationProp prop;
  CUmemAccessDesc accessDesc;

  struct Range {
    CUdeviceptr start;
    size_t sz;
  };

  std::vector<CUmemGenericAllocationHandle> handles_;
  std::vector<Range> va_ranges;
  std::vector<size_t> handleSizes_;

  size_t allocSz; // the size of the physical memory
  size_t reserveSz; // size of reserved VA

  size_t chunkSz; // allocation granularity

  Buffer(CUcontext context) : d_p(0ULL), prop(), allocSz(0ULL), reserveSz(0ULL), chunkSz(0ULL)
{
    CUdevice device;
    CUcontext prev_ctx;
    CUresult status = CUDA_SUCCESS;
    (void)status;

    status = cuCtxGetCurrent(&prev_ctx);
    assert(status == CUDA_SUCCESS);
    if (cuCtxSetCurrent(context) == CUDA_SUCCESS) {
        status = cuCtxGetDevice(&device);
        assert(status == CUDA_SUCCESS);
        status = cuCtxSetCurrent(prev_ctx);
        assert(status == CUDA_SUCCESS);
    }

    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = (int)device;
    prop.win32HandleMetaData = NULL;

    accessDesc.location = prop.location;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    status = cuMemGetAllocationGranularity(&chunkSz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    assert(status == CUDA_SUCCESS);
}

  // Reserves some extra space in order to speed up grow()
  CUresult reserve(size_t newSz) {
    CUresult status = CUDA_SUCCESS;
    CUdeviceptr newPtr = 0ULL;

    if (newSz <= reserveSz) {
      return CUDA_SUCCESS;
    }

    const size_t alignedSz = detail::round_up(newSz, chunkSz);

    // try to get new pointer right after what we had before
    status = cuMemAddressReserve(&newPtr, (alignedSz - reserveSz), 0ULL,
                                 d_p + reserveSz, 0ULL);

    // did not succeed
    if (status != CUDA_SUCCESS || (newPtr != d_p + reserveSz)) {
      // return the VA mapping to the driver
      if (newPtr != 0ULL) {
        (void)cuMemAddressFree(newPtr, (alignedSz - reserveSz));
      }
      // Find a new VA big enough for our needs
      status = cuMemAddressReserve(&newPtr, alignedSz, 0ULL, 0U, 0);
      if (status == CUDA_SUCCESS && d_p != 0ULL) {
        CUdeviceptr ptr = newPtr;
        // Found one, now unmap our previous allocations
        status = cuMemUnmap(d_p, allocSz);
        assert(status == CUDA_SUCCESS);
        for (size_t i = 0ULL; i < handles_.size(); i++) {
          const size_t hdl_sz = handleSizes_[i];
          // And remap them, enabling their access
          if ((status = cuBuffer(ptr, hdl_sz, 0ULL, handles_[i], 0ULL)) !=
              CUDA_SUCCESS)
            break;
          if ((status = cuMemSetAccess(ptr, hdl_sz, &accessDesc, 1ULL)) !=
              CUDA_SUCCESS)
            break;
          ptr += hdl_sz;
        }
        if (status != CUDA_SUCCESS) {
          // Failed the mapping somehow... clean up!
          status = cuMemUnmap(newPtr, alignedSz);
          assert(status == CUDA_SUCCESS);
          status = cuMemAddressFree(newPtr, alignedSz);
          assert(status == CUDA_SUCCESS);
        } else {
          // Clean up our old VA reservations!
          for (size_t i = 0ULL; i < va_ranges.size(); i++) {
            (void)cuMemAddressFree(va_ranges[i].start, va_ranges[i].sz);
          }
          va_ranges.clear();
        }
      }
      // record the new VA range we're using, so it can be released later
      if (status == CUDA_SUCCESS) {
        Range r;
        d_p = newPtr;
        reserveSz = alignedSz;
        r.start = newPtr;
        r.sz = alignedSz;
        va_ranges.push_back(r);
      }
    } else {
      Range r;
      r.start = newPtr;
      r.sz = alignedSz - reserveSz;
      va_ranges.push_back(r);
      if (d_p == 0ULL) {
        d_p = newPtr;
      }
      reserveSz = alignedSz;
    }

    return status;
  }

  // Actually commits num bytes of additional memory
  CUresult grow(size_t newSz) {
    CUresult status = CUDA_SUCCESS;
    CUmemGenericAllocationHandle handle;
    if (newSz <= allocSz) {
      return CUDA_SUCCESS;
    }

    const size_t size_diff = newSz - allocSz;
    // Round up to the next chunk size
    const size_t sz = detail::round_up(size_diff, chunkSz);

    if ((status = reserve(allocSz + sz)) != CUDA_SUCCESS) {
      return status;
    }

    if ((status = cuMemCreate(&handle, sz, &prop, 0)) == CUDA_SUCCESS) {
      if ((status = cuBuffer(d_p + allocSz, sz, 0ULL, handle, 0ULL)) ==
          CUDA_SUCCESS) {
        if ((status = cuMemSetAccess(d_p + allocSz, sz, &accessDesc, 1ULL)) ==
            CUDA_SUCCESS) {
          handles_.push_back(handle);
          handleSizes_.push_back(sz);
          allocSz += sz;
        }
        if (status != CUDA_SUCCESS) {
          (void)cuMemUnmap(d_p + allocSz, sz);
        }
      }
      if (status != CUDA_SUCCESS) {
        (void)cuMemRelease(handle);
      }
    }

    return status;
  }

};
#endif


} // namespace cmm

#undef CMM_OR_THROW