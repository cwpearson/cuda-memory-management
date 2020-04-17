#include <catch2/catch.hpp>

#include "cmm/cmm.hpp"
#include "writer.hpp"

TEMPLATE_TEST_CASE("managed", "[cuda][template]", int, double) {
    typedef cmm::Managed<TestType> Allocator;

    SECTION("ctor") {
        Allocator a;
        TestType *p = a.allocate(10);
        writer<<<10, 10>>>(p, 10);
        cudaDeviceSynchronize();
        a.deallocate(p, 10);
    }
}