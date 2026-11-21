# Exact per-sample monotone-hull check (#178). Host-only, SYCL-free.
add_library(fuzzybits_ir_pc_hull STATIC
  check.cc
  detail/bdd.cc
  detail/residual.cc
  detail/sat.cc
)
set_target_properties(fuzzybits_ir_pc_hull PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(fuzzybits_ir_pc_hull PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
target_link_libraries(fuzzybits_ir_pc_hull PUBLIC fuzzybits_ir_pc)

find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
  target_link_libraries(fuzzybits_ir_pc_hull PRIVATE OpenMP::OpenMP_CXX)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
