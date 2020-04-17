# CUDA Memory Management

C++ allocators for CUDA memory

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