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

  # Runs the fzb binary; the test itself links nothing from SCRAM.
  if(NOT SKBUILD)
    fuzzybits_add_imported_test(
      TARGET fuzzybits_compile_ftp_cli_tests
      SOURCES tests/cli_tests.cc
      LINK_LIBRARIES fuzzybits_compile_ftp
      ARGS "~[mc]"
    )
    target_compile_definitions(fuzzybits_compile_ftp_cli_tests PRIVATE FZB_PATH="$<TARGET_FILE:fzb>")
    add_dependencies(fuzzybits_compile_ftp_cli_tests fzb)
    set_tests_properties(fuzzybits_compile_ftp_cli_tests PROPERTIES TIMEOUT 300)
    if(FUZZYBITS_WITH_MC)
      add_test(NAME mc.compile_ftp.fzb_run COMMAND fuzzybits_compile_ftp_cli_tests "[mc]")
      set_tests_properties(mc.compile_ftp.fzb_run PROPERTIES LABELS "mc" TIMEOUT 60)
    endif()
  endif()
endif()
