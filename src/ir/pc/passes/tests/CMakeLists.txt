fuzzybits_ensure_catch2()

fuzzybits_add_imported_test(
  TARGET fuzzybits_ir_pc_passes_tests
  SOURCES passes_tests.cc
  LINK_LIBRARIES fuzzybits_ir_pc_passes
)

fuzzybits_add_imported_test(
  TARGET fuzzybits_ir_pc_passes_scale_tests
  SOURCES scale_tests.cc
  LINK_LIBRARIES fuzzybits_ir_pc_passes
)

# End-to-end through the FTP reader and the .pcfb writer; reads
# FUZZYBITS_PRIVATE_FIXTURES_DIR when set and prints aggregate counts only.
fuzzybits_add_imported_test(
  TARGET fuzzybits_ir_pc_passes_io_tests
  SOURCES io_tests.cc
  LINK_LIBRARIES fuzzybits_ir_pc_passes fuzzybits_compile_ftp fuzzybits_io_pcfb fuzzybits_exact_eval pcfb
)

# Known bugs (ctest -L bug): these fail until their issues are fixed. [crash] cases read out of bounds
# today, so they run as a ctest entry of their own.
fuzzybits_add_imported_test(
  TARGET fuzzybits_ir_pc_passes_bug_tests
  SOURCES bug_tests.cc
  LINK_LIBRARIES fuzzybits_ir_pc_passes
  LABELS bug
  ARGS "~[crash]"
)
add_test(NAME fuzzybits_ir_pc_passes_bug_tests.crash COMMAND fuzzybits_ir_pc_passes_bug_tests "[crash]")
set_tests_properties(fuzzybits_ir_pc_passes_bug_tests.crash PROPERTIES LABELS bug)
