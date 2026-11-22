# ir::pra -> ir::pc lowering under a quantification config. SCRAM-free: links
# only the IR, the API types and the .pcfb writer (for its option types).
add_library(fuzzybits_compile_pra_to_pc STATIC
  lower.cpp
  write_options.cpp
  descriptor.h
)
set_target_properties(fuzzybits_compile_pra_to_pc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(fuzzybits_compile_pra_to_pc PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
target_link_libraries(fuzzybits_compile_pra_to_pc
  PUBLIC fuzzybits_ir_pra fuzzybits_ir_pc fuzzybits_io_pcfb
)

if(BUILD_TESTING)
  fuzzybits_ensure_catch2()
  fuzzybits_add_imported_test(
    TARGET fuzzybits_compile_pra_to_pc_tests
    SOURCES tests/lower_tests.cc tests/projection_tests.cc tests/pcfb_tests.cc tests/scale_tests.cc tests/literal_tests.cc
    LINK_LIBRARIES fuzzybits_compile_pra_to_pc fuzzybits_exact_eval pcfb
  )
endif()
