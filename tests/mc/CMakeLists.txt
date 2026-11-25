# Monte Carlo tests. Label `mc`. Every MC invocation has TIMEOUT 60 so a
# non-terminating sampler (#22) fails fast. Device selection is via
# ACPP_VISIBILITY_MASK (test presets: omp on CPU, cuda/metal on GPU).

if(NOT FUZZYBITS_WITH_MC)
  return()
endif()

# add_sycl_to_target needs AdaptiveCpp::acpp-rt in this directory scope;
# src/mc's find_package is not visible to the tests/ tree.
find_package(_adaptivecpp)

set(MC_TEST_TIMEOUT 60)

add_executable(mc_cli_correctness_tests cli_correctness_tests.cc)
# Catch2WithMain must precede mc: libmc PUBLIC-links ${LIBS}, which still
# contains Boost::unit_test_framework and would otherwise own main.
target_link_libraries(mc_cli_correctness_tests PRIVATE Catch2::Catch2WithMain ${SCRAM_TEST_LIBS} scram mc)
target_compile_definitions(mc_cli_correctness_tests PRIVATE
  SCRAM_CLI_PATH="$<TARGET_FILE:scram-cli>"
  PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
target_include_directories(mc_cli_correctness_tests PRIVATE
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/tests"
)
add_dependencies(mc_cli_correctness_tests scram-cli)

foreach(_model chinese das9201 isp9601 baobab2 small_tree bscu kn_synthetic)
  add_test(NAME mc.cli.${_model} COMMAND mc_cli_correctness_tests "[${_model}]")
  set_tests_properties(mc.cli.${_model} PROPERTIES
    LABELS "mc"
    TIMEOUT ${MC_TEST_TIMEOUT}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  )
endforeach()

add_executable(mc_fzb_cli_identity_tests fzb_cli_identity_tests.cc)
target_link_libraries(mc_fzb_cli_identity_tests PRIVATE Catch2::Catch2WithMain)
target_compile_definitions(mc_fzb_cli_identity_tests PRIVATE
  SCRAM_CLI_PATH="$<TARGET_FILE:scram-cli>"
  FZB_PATH="$<TARGET_FILE:fzb>"
  PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
add_dependencies(mc_fzb_cli_identity_tests scram-cli fzb)
foreach(_case n65536 n2p24 seed smalltree bscu)
  add_test(NAME mc.fzb_cli_identity.${_case} COMMAND mc_fzb_cli_identity_tests "[${_case}]")
  set_tests_properties(mc.fzb_cli_identity.${_case} PROPERTIES
    LABELS "mc"
    TIMEOUT ${MC_TEST_TIMEOUT}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  )
endforeach()

add_executable(mc_progress_log_cli_tests progress_log_cli_tests.cc)
target_link_libraries(mc_progress_log_cli_tests PRIVATE Catch2::Catch2WithMain)
target_include_directories(mc_progress_log_cli_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(mc_progress_log_cli_tests PRIVATE
  FZB_PATH="$<TARGET_FILE:fzb>"
  PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
add_dependencies(mc_progress_log_cli_tests fzb)
add_test(NAME mc.progress_log_cli COMMAND mc_progress_log_cli_tests)
set_tests_properties(mc.progress_log_cli PROPERTIES
  LABELS "mc"
  TIMEOUT ${MC_TEST_TIMEOUT}
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)

add_executable(mc_bitpack_identity_tests bitpack_identity_tests.cc)
target_link_libraries(mc_bitpack_identity_tests PRIVATE Catch2::Catch2WithMain mc)
target_include_directories(mc_bitpack_identity_tests PRIVATE
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/tests"
)
add_sycl_to_target(TARGET mc_bitpack_identity_tests SOURCES bitpack_identity_tests.cc)
add_test(NAME mc.bitpack_identity COMMAND mc_bitpack_identity_tests)
set_tests_properties(mc.bitpack_identity PROPERTIES
  LABELS "mc"
  TIMEOUT ${MC_TEST_TIMEOUT}
)

# Host-only: the bit-sliced Bernoulli sampler against Philox known answers, exact references and
# statistical checks. Device output is compared with the same reference by mc.bitpack_identity.
add_executable(mc_bernoulli_sampler_tests bernoulli_sampler_tests.cc)
target_link_libraries(mc_bernoulli_sampler_tests PRIVATE Catch2::Catch2WithMain mc)
target_include_directories(mc_bernoulli_sampler_tests PRIVATE
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/tests"
)
add_test(NAME mc.bernoulli_sampler COMMAND mc_bernoulli_sampler_tests)
set_tests_properties(mc.bernoulli_sampler PROPERTIES
  LABELS "mc"
  TIMEOUT ${MC_TEST_TIMEOUT}
)

add_executable(mc_cardinality_gate_tests cardinality_gate_tests.cc)
target_link_libraries(mc_cardinality_gate_tests PRIVATE Catch2::Catch2WithMain mc)
target_include_directories(mc_cardinality_gate_tests PRIVATE
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/tests"
)
add_sycl_to_target(TARGET mc_cardinality_gate_tests SOURCES cardinality_gate_tests.cc)
add_test(NAME mc.cardinality_gate COMMAND mc_cardinality_gate_tests)
# 80 kernels (4 widths x 10 forms and slot counts x 2 lane forms) are JIT-compiled on first launch, and run
# in both work-group orders.
set_tests_properties(mc.cardinality_gate PROPERTIES
  LABELS "mc"
  TIMEOUT 600
)

add_executable(mc_gate_kernel_tests gate_kernel_tests.cc)
target_link_libraries(mc_gate_kernel_tests PRIVATE Catch2::Catch2WithMain mc)
target_include_directories(mc_gate_kernel_tests PRIVATE
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/tests"
)
add_sycl_to_target(TARGET mc_gate_kernel_tests SOURCES gate_kernel_tests.cc)
add_test(NAME mc.gate_kernel COMMAND mc_gate_kernel_tests)
# About 100 kernels (connectives x slot counts x lane forms x widths) are JIT-compiled on first launch.
set_tests_properties(mc.gate_kernel PROPERTIES
  LABELS "mc"
  TIMEOUT 600
)

add_executable(mc_tally_kernel_tests tally_kernel_tests.cc)
target_link_libraries(mc_tally_kernel_tests PRIVATE Catch2::Catch2WithMain mc)
target_include_directories(mc_tally_kernel_tests PRIVATE
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/tests"
)
add_sycl_to_target(TARGET mc_tally_kernel_tests SOURCES tally_kernel_tests.cc)
add_test(NAME mc.tally_kernel COMMAND mc_tally_kernel_tests)
set_tests_properties(mc.tally_kernel PROPERTIES
  LABELS "mc"
  TIMEOUT ${MC_TEST_TIMEOUT}
)

# 128- and 256-bit bitpacks must be rejected at compile time, and only for the
# stated reason: the cardinality kernel static_asserts on width, and
# sycl::popcount is constrained to the standard integer types. Each
# test builds an EXCLUDE_FROM_ALL object target and passes iff that build fails
# and every compiler error matches the expected diagnostic.
# Locations are file:line:col (clang) or file(line,col) (clang-cl).
set(_compile_fail_loc "[:(][0-9]+[:,][0-9]+\\)?: error: ")
set(_compile_fail_regex_cardinality "cardinality\\.h${_compile_fail_loc}static assertion failed due to requirement '.*kSupported'.*cardinality gates need an unsigned bitpack of at most 64 bits")
set(_compile_fail_regex_tally "tally\\.h${_compile_fail_loc}no matching function for call to 'popcount'")
foreach(_kernel cardinality tally)
  string(TOUPPER "${_kernel}" _kernel_upper)
  foreach(_bits 128 256)
    set(_target mc_compile_fail_${_kernel}_u${_bits})
    add_library(${_target} OBJECT EXCLUDE_FROM_ALL bitpack_width_compile_fail.cc)
    target_link_libraries(${_target} PRIVATE mc)
    target_include_directories(${_target} PRIVATE
      "${PROJECT_SOURCE_DIR}/src"
      "${PROJECT_SOURCE_DIR}/tests"
    )
    target_compile_definitions(${_target} PRIVATE FZB_BITS=${_bits} FZB_KERNEL_${_kernel_upper})
    add_sycl_to_target(TARGET ${_target} SOURCES bitpack_width_compile_fail.cc)
    add_test(NAME mc.compile_fail.${_kernel}.u${_bits}
      COMMAND ${CMAKE_COMMAND}
        "-DBUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DTARGET=${_target}"
        "-DCONFIG=$<CONFIG>"
        "-DEXPECT=${_compile_fail_regex_${_kernel}}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_compile_failure.cmake")
    set_tests_properties(mc.compile_fail.${_kernel}.u${_bits} PROPERTIES
      LABELS "mc;compile-fail"
      RESOURCE_LOCK fuzzybits_build_tree
      TIMEOUT 300
    )
  endforeach()
endforeach()

add_executable(mc_device_selection_tests device_selection_tests.cc)
target_link_libraries(mc_device_selection_tests PRIVATE Catch2::Catch2WithMain mc fuzzybits_runtime)
target_include_directories(mc_device_selection_tests PRIVATE
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/tests"
)
add_sycl_to_target(TARGET mc_device_selection_tests SOURCES device_selection_tests.cc)
add_test(NAME mc.device_selection COMMAND mc_device_selection_tests "~[late-mask]")
set_tests_properties(mc.device_selection PROPERTIES
  LABELS "mc"
  TIMEOUT ${MC_TEST_TIMEOUT}
)
# Starts with no mask so AdaptiveCpp loads every backend, as for an fzb user.
# Catch2 exits 4 when every test case is skipped (only one backend present).
add_test(NAME mc.device_selection.late_mask COMMAND mc_device_selection_tests "[late-mask]")
set_tests_properties(mc.device_selection.late_mask PROPERTIES
  LABELS "mc"
  TIMEOUT ${MC_TEST_TIMEOUT}
  ENVIRONMENT_MODIFICATION "ACPP_VISIBILITY_MASK=unset:"
  SKIP_RETURN_CODE 4
)
