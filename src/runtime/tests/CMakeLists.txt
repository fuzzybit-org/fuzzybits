fuzzybits_ensure_catch2()
fuzzybits_add_imported_test(
  TARGET fzb_runtime_tests
  SOURCES runtime_tests.cc session_metal_tests.cc progress_log_tests.cc pcfb_constant_tests.cc
  LINK_LIBRARIES fuzzybits_runtime fuzzybits_compile fuzzybits_progress_log
  ARGS "~[issue-171]"
)
set_tests_properties(fzb_runtime_tests PROPERTIES
  LABELS "mc"
  TIMEOUT 120
)

# Session::step(n) on Metal differs from n single steps (#171). Its own entry
# so the gating Metal CI step can exclude it by name. Catch2 exits 4 when the
# case is skipped (no Metal device).
add_test(NAME fzb_runtime_tests.metal_step_n COMMAND fzb_runtime_tests "[issue-171]")
set_tests_properties(fzb_runtime_tests.metal_step_n PROPERTIES
  LABELS "mc"
  TIMEOUT 120
  SKIP_RETURN_CODE 4
)

# Known bugs of the .pcfb loader (ctest -L bug): these fail until their issues are fixed.
fuzzybits_add_imported_test(
  TARGET fzb_runtime_pcfb_bug_tests
  SOURCES pcfb_bug_tests.cc
  LINK_LIBRARIES fuzzybits_runtime fuzzybits_compile
  LABELS bug mc
  ARGS "~[crash]"
)
set_tests_properties(fzb_runtime_pcfb_bug_tests PROPERTIES TIMEOUT 120)
# [crash] cases end the process today (SIGSEGV), so they run as a ctest entry of their own.
add_test(NAME fzb_runtime_pcfb_bug_tests.crash COMMAND fzb_runtime_pcfb_bug_tests "[crash]")
set_tests_properties(fzb_runtime_pcfb_bug_tests.crash PROPERTIES LABELS "bug;mc" TIMEOUT 300)
