# ir::pc ↔ .pcfb core-1.1 through the pinned pcfb::builder and reader. SCRAM-free: the
# library links only the IR, the API types and pcfb (cmake/FuzzybitsPcfb.cmake
# defines that target after src/io; link names resolve at generate time).
add_library(fuzzybits_io_pcfb STATIC
  write.cpp
  read.cpp
  detail/order.cpp
  detail/writer.cpp
  detail/reader.cpp
  descriptor.h
  detail/model.h
)
set_target_properties(fuzzybits_io_pcfb PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(fuzzybits_io_pcfb PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
target_link_libraries(fuzzybits_io_pcfb
  PUBLIC fuzzybits_io fuzzybits_ir_pc
  PRIVATE pcfb
)

# Same general.generator_version as the nombrilo path, without its SCRAM-derived
# version.h.
set(_fuzzybits_pcfb_generator_version "${PROJECT_VERSION}")
if(SCRAM_GIT_REVISION AND NOT SCRAM_GIT_REVISION STREQUAL "unknown")
  string(APPEND _fuzzybits_pcfb_generator_version "+g${SCRAM_GIT_REVISION}")
endif()
target_compile_definitions(fuzzybits_io_pcfb PRIVATE
  FUZZYBITS_PCFB_GENERATOR_VERSION="${_fuzzybits_pcfb_generator_version}"
)
unset(_fuzzybits_pcfb_generator_version)

if(BUILD_TESTING)
  fuzzybits_ensure_catch2()
  fuzzybits_add_imported_test(
    TARGET fuzzybits_io_pcfb_tests
    SOURCES tests/write_tests.cc tests/kind_tests.cc tests/read_tests.cc
    LINK_LIBRARIES fuzzybits_io_pcfb fuzzybits_compile_ftp fuzzybits_exact_eval pcfb
  )

  fuzzybits_add_imported_test(
    TARGET fuzzybits_io_pcfb_read_scale_tests
    SOURCES tests/read_scale_tests.cc
    LINK_LIBRARIES fuzzybits_io_pcfb pcfb
  )

  # Query-kind guard (#217): header-only check plus the exact evaluator; needs only pcfb.
  fuzzybits_add_imported_test(
    TARGET fuzzybits_io_pcfb_guard_tests
    SOURCES tests/query_guard_tests.cc
    LINK_LIBRARIES fuzzybits_exact_eval pcfb
  )

  # Runs the fzb binary on written files; the test itself links nothing from SCRAM.
  if(NOT SKBUILD)
    fuzzybits_add_imported_test(
      TARGET fuzzybits_io_pcfb_cli_tests
      SOURCES tests/cli_tests.cc
      LINK_LIBRARIES fuzzybits_io_pcfb fuzzybits_compile_ftp pcfb
      ARGS "~[mc]"
    )
    target_compile_definitions(fuzzybits_io_pcfb_cli_tests PRIVATE FZB_PATH="$<TARGET_FILE:fzb>")
    add_dependencies(fuzzybits_io_pcfb_cli_tests fzb)
    if(FUZZYBITS_WITH_MC)
      add_test(NAME mc.io_pcfb.fzb_run COMMAND fuzzybits_io_pcfb_cli_tests "[mc]")
      set_tests_properties(mc.io_pcfb.fzb_run PROPERTIES
        LABELS "mc"
        TIMEOUT 60
      )
    endif()
  endif()
endif()
