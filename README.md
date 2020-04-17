# CUDA Memory Management

[![Build Status](https://travis-ci.com/cwpearson/cuda-memory-management.svg?branch=master)](https://travis-ci.com/cwpearson/cuda-memory-management)

C++11 allocators for CUDA memory

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

```c++
#include "cmm/cmm.hpp"

cmm::Managed<int> allocator;
int *p = allocator.allocate(10);
allocator.deallocate(p, 10);
```

## Notes

- [ ] bind allocators to CUDA devices (Allocators do not have to be stateless starting with C++11)
- [ ] A [low-level device allocator](https://devblogs.nvidia.com/introducing-low-level-gpu-virtual-memory-management/)

