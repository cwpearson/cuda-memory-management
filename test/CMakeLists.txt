add_executable(test-all test_main.cu
  test_malloc.cu
  test_zerocopy.cu
  test_roundup.cu
  test_zerocopy.cu
  test_std_vector.cu
)

target_include_directories(test-all SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)
target_link_libraries(test-all cmm)
add_test(NAME test_all COMMAND cuda-memcheck test-all -a)


