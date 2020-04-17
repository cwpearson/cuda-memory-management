#include <catch2/catch.hpp>

#include "cmm/cmm.hpp"

TEST_CASE("roundup", "") {

    using namespace cmm::detail;

    SECTION("ctor") {
        REQUIRE(0 == round_up(0, 1));
        REQUIRE(10 == round_up(10, 10));
        REQUIRE(10 == round_up(10, 1));
    }
}