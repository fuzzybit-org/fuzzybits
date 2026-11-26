add_library(fuzzybits_io_ftp STATIC
  read.cpp
  summary.cpp
  write.cpp
  detail/parser.cpp
  detail/pra_builder.cpp
  detail/writer.cpp
  descriptor.h
  summary.h
  detail/model.h
)
set_target_properties(fuzzybits_io_ftp PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(fuzzybits_io_ftp PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
target_link_libraries(fuzzybits_io_ftp
  PUBLIC fuzzybits_io
  PRIVATE fuzzybits_ir_pc fuzzybits_ir_pra
)

if(BUILD_TESTING)
  fuzzybits_ensure_catch2()
  fuzzybits_add_imported_test(
    TARGET fuzzybits_io_ftp_tests
    SOURCES tests/write_tests.cc tests/project_tests.cc tests/conformance_tests.cc tests/grammar_tests.cc
    LINK_LIBRARIES fuzzybits_io_ftp fuzzybits_ir_pra fuzzybits_compile_ftp
  )
endif()

# Known bugs (ctest -L bug): these fail until their issues are fixed.
if(BUILD_TESTING)
  fuzzybits_add_imported_test(
    TARGET fuzzybits_io_ftp_bug_tests
    SOURCES tests/bug_tests.cc
    LINK_LIBRARIES fuzzybits_io_ftp fuzzybits_ir_pra
    LABELS bug
  )
endif()
