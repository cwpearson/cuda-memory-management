# CUDA Memory Management

[![Build Status](https://travis-ci.com/cwpearson/cuda-memory-management.svg?branch=master)](https://travis-ci.com/cwpearson/cuda-memory-management)

C++11 allocators for CUDA memory for
* device memory `cmm::Malloc`
* zero-copy memory `cmm::ZeroCopy`
* managed memory `cmm::Managed`

## Quick Start

Copy `cmm/cmm.hpp` to your project.

Use an allocator

```c++
#include "cmm/cmm.hpp"

cmm::ZeroCopy<int> allocator;
int *p = allocator.allocate(10);
allocator.deallocate(p, 10);
```

```c++
#include "cmm/cmm.hpp"

cmm::Malloc<int> allocator;
int *p = allocator.allocate(10);
allocator.deallocate(p, 10);
```

```c++
#include "cmm/cmm.hpp"

cmm::Managed<int> allocator;
int *p = allocator.allocate(10);
allocator.deallocate(p, 10);
```

Allocators can also be assigned to a specific device.
Otherwise, they will respect calls to `cudaSetDevice()`.

```c++
#include "cmm/cmm.hpp"

cmm::Managed<int> allocator(3); // gpu 3
int *p = allocator.allocate(10);
allocator.deallocate(p, 10);
```

## Running Tests

```
make && make CTEST_OUTPUT_ON_FAILURE=1 test
```

## Roadmap

- [ ] Allow passing of flags to `cudaHostAlloc` in `cmm::ZeroCopy`
- [ ] A [low-level device allocator](https://devblogs.nvidia.com/introducing-low-level-gpu-virtual-memory-management/)
