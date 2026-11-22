fuzzybits_ensure_catch2()
fuzzybits_add_imported_test(
  TARGET fuzzybits_progress_line_tests
  SOURCES progress_line_tests.cc
  LINK_LIBRARIES fuzzybits_progress_line
)
