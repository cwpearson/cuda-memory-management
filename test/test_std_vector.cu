#include <vector>

#include <catch2/catch.hpp>

#include "cmm/cmm.hpp"

inline int device_count() {
  int ret;
  cudaGetDeviceCount(&ret);
  return ret;
}

TEMPLATE_TEST_CASE("std::vector/Malloc", "[cuda][template]", int, double) {
  typedef cmm::Malloc<TestType> Allocator;

  SECTION("default") { auto v = std::vector<TestType, Allocator>(Allocator()); }

  // can't use any ctor that writes to GPU memory
}

TEMPLATE_TEST_CASE("std::vector/Managed", "[cuda][template]", int, double) {
  typedef cmm::Managed<TestType> Allocator;

  SECTION("default") { auto v = std::vector<TestType, Allocator>(Allocator()); }

  SECTION("default assigned") {
    for (int d = 0; d < device_count(); ++d) {
      auto v = std::vector<TestType, Allocator>(Allocator(d));
    }
  }

  SECTION("fill") {
    auto v = std::vector<TestType, Allocator>(10, 0, Allocator());
  }

  SECTION("fill assigned") {
    for (int d = 0; d < device_count(); ++d) {
        auto v = std::vector<TestType, Allocator>(10, 0, Allocator(d));
    }
  }

}

TEMPLATE_TEST_CASE("std::vector/ZeroCopy", "[cuda][template]", int, double) {
  typedef cmm::ZeroCopy<TestType> Allocator;

  SECTION("default") { auto v = std::vector<TestType, Allocator>(Allocator()); }

  SECTION("default assigned") {
    for (int d = 0; d < device_count(); ++d) {
      auto v = std::vector<TestType, Allocator>(Allocator(d));
    }
  }

  SECTION("fill") {
    auto v = std::vector<TestType, Allocator>(10, 0, Allocator());
  }

  SECTION("fill assigned") {
    for (int d = 0; d < device_count(); ++d) {
        auto v = std::vector<TestType, Allocator>(10, 0, Allocator(d));
    }
  }
}