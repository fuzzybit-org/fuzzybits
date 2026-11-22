fuzzybits_ensure_catch2()

fuzzybits_add_imported_test(
  TARGET fuzzybits_ir_pc_hull_tests
  SOURCES hull_check_tests.cc
  LINK_LIBRARIES fuzzybits_ir_pc_hull
)
set_tests_properties(fuzzybits_ir_pc_hull_tests PROPERTIES TIMEOUT 600)

fuzzybits_add_imported_test(
  TARGET fuzzybits_ir_pc_hull_conformance_tests
  SOURCES hull_conformance_tests.cc
  LINK_LIBRARIES fuzzybits_ir_pc_hull fuzzybits_io_ftp
)
set_tests_properties(fuzzybits_ir_pc_hull_conformance_tests PROPERTIES TIMEOUT 600)

# Throughput on a synthetic circuit (and optionally an FTP model); not a test.
add_executable(fuzzybits_ir_pc_hull_bench hull_bench.cc)
target_link_libraries(fuzzybits_ir_pc_hull_bench PRIVATE fuzzybits_ir_pc_hull fuzzybits_io_ftp)
if(OpenMP_CXX_FOUND)
  target_link_libraries(fuzzybits_ir_pc_hull_bench PRIVATE OpenMP::OpenMP_CXX)
endif()
