#include <catch2/catch.hpp>

#include "cmm/cmm.hpp"
#include "writer.hpp"

TEMPLATE_TEST_CASE("zerocopy", "[cuda][template]", int, double) {
    typedef cmm::ZeroCopy<TestType> Allocator;

    SECTION("ctor") {
        Allocator a;
        TestType *p = a.allocate(10);
        writer<<<10, 10>>>(p, 10);
        cudaDeviceSynchronize();
        a.deallocate(p, 10);
    }
}