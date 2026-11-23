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
  LINK_LIBRARIES fuzzybits_ir_pc_hull fuzzybits_io_ftp fuzzybits_compile_ftp
)
set_tests_properties(fuzzybits_ir_pc_hull_conformance_tests PROPERTIES TIMEOUT 600)

fuzzybits_add_imported_test(
  TARGET fuzzybits_ir_pc_hull_exact_tests
  SOURCES hull_exact_tests.cc
  LINK_LIBRARIES fuzzybits_ir_pc_hull fuzzybits_io_pcfb fuzzybits_io_ftp fuzzybits_compile_ftp
)
set_tests_properties(fuzzybits_ir_pc_hull_exact_tests PROPERTIES TIMEOUT 600)

# Throughput on a synthetic circuit (and optionally an FTP model); not a test.
add_executable(fuzzybits_ir_pc_hull_bench hull_bench.cc)
target_link_libraries(fuzzybits_ir_pc_hull_bench PRIVATE fuzzybits_ir_pc_hull fuzzybits_io_ftp fuzzybits_compile_ftp)
if(OpenMP_CXX_FOUND)
  target_link_libraries(fuzzybits_ir_pc_hull_bench PRIVATE OpenMP::OpenMP_CXX)
endif()

# The decomposition by initiator decides every sampled lane like the initiator rule as lowered.
function(fuzzybits_hull_bench_initiator_test NAME)
  cmake_parse_arguments(PARSE_ARGV 1 HB "" "" "FILES;TARGETS")
  set(args)
  foreach(file IN LISTS HB_FILES)
    list(APPEND args --ftp ${PROJECT_SOURCE_DIR}/input/ftp/conformance/${file})
  endforeach()
  foreach(target IN LISTS HB_TARGETS)
    list(APPEND args --target ${target})
  endforeach()
  foreach(level IN ITEMS 0 1)
    set(test fuzzybits_ir_pc_hull_bench.initiators.${NAME}.c${level})
    add_test(NAME ${test}
      COMMAND fuzzybits_ir_pc_hull_bench ${args} --hull --max-initiators 1 --level ${level}
        --lanes 65536 --seed 1 --decompose-initiators --verify-decomposition)
    set_tests_properties(${test} PROPERTIES TIMEOUT 120 LABELS unit)
  endforeach()
endfunction()

fuzzybits_hull_bench_initiator_test(initiator_rule
  FILES initiator_rule/model.ftp TARGETS TOP SEQ-1 SEQ-3 SEQ-4)
fuzzybits_hull_bench_initiator_test(success_branch_delete_term
  FILES success_branch_delete_term/model.ftp TARGETS TOP SEQ-2 SEQ-3)
fuzzybits_hull_bench_initiator_test(delete_term_flags
  FILES delete_term_flags/model.ftp)
fuzzybits_hull_bench_initiator_test(imports_tags_duplicates
  FILES imports_tags_duplicates/model.ftp)
fuzzybits_hull_bench_initiator_test(multi_file_project
  FILES multi_file_project/01_top.ftp multi_file_project/02_systems.ftp multi_file_project/03_imports.ftp)
