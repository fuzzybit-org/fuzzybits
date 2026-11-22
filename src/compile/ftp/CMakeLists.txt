# CAFTA FTP project -> .pcfb: reader, quantification config, ir::pra -> ir::pc
# lowering, ir::pc passes and the writer. SCRAM-free; fzb and the Python module
# link it next to the MEF path in fuzzybits_compile.
add_library(fuzzybits_compile_ftp STATIC
  compile.cpp
  json.cpp
  compile.h
  json.h
)
set_target_properties(fuzzybits_compile_ftp PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(fuzzybits_compile_ftp PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
target_link_libraries(fuzzybits_compile_ftp
  PUBLIC
    fuzzybits_compile_pra_to_pc
    fuzzybits_io_ftp
    fuzzybits_ir_pc_passes
    fuzzybits_io_pcfb
    fuzzybits_sha256
)

if(BUILD_TESTING)
  fuzzybits_ensure_catch2()
  fuzzybits_add_imported_test(
    TARGET fuzzybits_compile_ftp_tests
    SOURCES tests/compile_tests.cc tests/json_tests.cc
    LINK_LIBRARIES fuzzybits_compile_ftp fuzzybits_exact_eval pcfb
  )
endif()
