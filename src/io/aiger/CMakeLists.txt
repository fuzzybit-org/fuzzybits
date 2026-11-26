add_library(fuzzybits_io_aiger STATIC
  read.cpp
  write.cpp
  detail/parser.cpp
  detail/builder.cpp
  detail/writer.cpp
  descriptor.h
  detail/model.h
)
set_target_properties(fuzzybits_io_aiger PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(fuzzybits_io_aiger PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
target_link_libraries(fuzzybits_io_aiger
  PUBLIC fuzzybits_io
  PRIVATE fuzzybits_ir_pc
)

if(BUILD_TESTING)
  fuzzybits_ensure_catch2()
  fuzzybits_add_imported_test(
    TARGET fuzzybits_io_aiger_tests
    SOURCES tests/read_tests.cc tests/write_tests.cc
    LINK_LIBRARIES fuzzybits_io_aiger
  )
endif()

# Known bugs (ctest -L bug): these fail until their issues are fixed. [crash] cases terminate the
# process today, so they run as a ctest entry of their own and cannot hide the other cases.
if(BUILD_TESTING)
  fuzzybits_add_imported_test(
    TARGET fuzzybits_io_aiger_bug_tests
    SOURCES tests/bug_tests.cc
    LINK_LIBRARIES fuzzybits_io_aiger
    LABELS bug
    ARGS "~[crash]"
  )
  add_test(NAME fuzzybits_io_aiger_bug_tests.crash COMMAND fuzzybits_io_aiger_bug_tests "[crash]")
  set_tests_properties(fuzzybits_io_aiger_bug_tests.crash PROPERTIES LABELS bug)
endif()
