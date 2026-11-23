add_library(fuzzybits_io INTERFACE)
target_link_libraries(fuzzybits_io INTERFACE fuzzybits_api fuzzybits_ir)

add_subdirectory(ftp)
add_subdirectory(aiger)
add_subdirectory(pcfb)

if(BUILD_TESTING)
  fuzzybits_add_imported_test(
    TARGET fuzzybits_io_charconv_tests
    SOURCES tests/charconv_tests.cc
    LINK_LIBRARIES fuzzybits_io_ftp
  )
endif()
