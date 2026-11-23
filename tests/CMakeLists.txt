# SPDX-License-Identifier: GPL-3.0-or-later
# vim: set foldmarker={{{,}}} foldlevel=0 foldmethod=marker:

# Catch2 is fetched once at the top level (cmake/FuzzybitsCatch2.cmake, v3.4.0).
# This directory is the SCRAM/MC suite; ir::pc, FTP/AIGER, and fzb compile
# tests live next to those targets. Host-only `fzb compile` sets
# FUZZYBITS_WITH_MC=OFF and has no libscram/libmc to link here.
if(NOT FUZZYBITS_WITH_MC)
  message(STATUS "tests/: skipping SCRAM/MC suite (FUZZYBITS_WITH_MC=OFF)")
  return()
endif()

include(Catch)

include_directories("${CMAKE_SOURCE_DIR}/src")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# ---------------------------------------------------------------------------
# Legacy Catch2 suite (SCRAM core / BDD). pi/mocus/zbdd re-entry is deferred:
# Catch2 v3 Session::run is once-per-process. Override via SCRAM_TEST_ALGORITHM.
# ---------------------------------------------------------------------------
set(SCRAM_CORE_TEST_SOURCE
  version_tests.cc
  linear_map_tests.cc
  linear_set_tests.cc
  xml_stream_tests.cc
  settings_tests.cc
  project_tests.cc
  element_tests.cc
  event_tests.cc
  expression_tests.cc
  ccf_group_tests.cc
  fault_tree_tests.cc
  alignment_tests.cc
  pdag_tests.cc
  initializer_tests.cc
  risk_analysis_tests.cc
  bench_core_tests.cc
  bench_two_train_tests.cc
  bench_lift_tests.cc
  bench_ne574_tests.cc
  bench_theatre_tests.cc
  bench_three_motor_tests.cc
  bench_200_event_tests.cc
  bench_small_tree_tests.cc
  bench_bscu_tests.cc
  bench_chinese_tree_tests.cc
  bench_baobab1_tests.cc
  bench_baobab2_tests.cc
  bench_CEA9601_tests.cc
  bench_hipps_tests.cc
  bench_attack.cc
  bench_gas_leak.cc
  performance_tests.cc
)

configure_file(scram_unit_test_driver.cc.in scram_unit_test_driver.cc @ONLY)

add_executable(scram_legacy_tests
  ${CMAKE_CURRENT_BINARY_DIR}/scram_unit_test_driver.cc
  ${SCRAM_CORE_TEST_SOURCE}
)
# ${LIBS} includes Boost::unit_test_framework (a leftover from the Boost.Test
# era) which defines main; drop it so Catch2 can own the entry point.
# libscram.so has undefined DirectEval symbols that live in libmc.so.
set(SCRAM_TEST_LIBS ${LIBS})
list(REMOVE_ITEM SCRAM_TEST_LIBS Boost::unit_test_framework)

target_link_libraries(scram_legacy_tests PRIVATE ${SCRAM_TEST_LIBS} scram mc Catch2::Catch2)
target_compile_options(scram_legacy_tests PRIVATE $<$<CONFIG:DEBUG>:${SCRAM_CXX_FLAGS_DEBUG}>)
target_include_directories(scram_legacy_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

add_test(NAME scram_legacy_tests COMMAND scram_legacy_tests)
set_tests_properties(scram_legacy_tests PROPERTIES
  LABELS "legacy"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  TIMEOUT 600
)

install(
  TARGETS scram_legacy_tests
  RUNTIME DESTINATION bin
  COMPONENT testing
)

# ---------------------------------------------------------------------------
# Monte Carlo correctness / bit-identity / device-selection tests.
# ---------------------------------------------------------------------------
add_subdirectory(mc)

# ---------------------------------------------------------------------------
# Python tests are owned by issue #32. Reserve the CTest label so that suite
# can `set_tests_properties(... LABELS python)` without colliding.
# ---------------------------------------------------------------------------
# add_test(NAME python.placeholder COMMAND ${CMAKE_COMMAND} -E true)
# set_tests_properties(python.placeholder PROPERTIES LABELS python)
