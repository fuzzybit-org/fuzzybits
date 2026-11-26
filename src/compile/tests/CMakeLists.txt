fuzzybits_ensure_catch2()
fuzzybits_add_imported_test(
  TARGET fzb_compile_tests
  SOURCES compile_tests.cc
  LINK_LIBRARIES fuzzybits_compile
)
if(NOT APPLE AND NOT FUZZYBITS_WITH_MC)
  target_sources(fzb_compile_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/compile/direct_eval_host_stubs.cc")
  target_include_directories(fzb_compile_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/mc")
elseif(FUZZYBITS_WITH_MC)
  target_link_libraries(fzb_compile_tests PRIVATE mc)
endif()
# Aralia models are large enough that exhaustive eval of chinese (~25 events)
# needs more than Catch's default 5s.
set_tests_properties(fzb_compile_tests PROPERTIES TIMEOUT 120)

fuzzybits_add_imported_test(
  TARGET fzb_nombrilo_tests
  SOURCES nombrilo_tests.cc
  LINK_LIBRARIES fuzzybits_compile
)
if(NOT APPLE AND NOT FUZZYBITS_WITH_MC)
  target_sources(fzb_nombrilo_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/compile/direct_eval_host_stubs.cc")
  target_include_directories(fzb_nombrilo_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/mc")
elseif(FUZZYBITS_WITH_MC)
  target_link_libraries(fzb_nombrilo_tests PRIVATE mc)
endif()
set_tests_properties(fzb_nombrilo_tests PROPERTIES TIMEOUT 600)

# Known nombrilo bugs (ctest -L bug): these fail until their issues are fixed. [crash] cases terminate the
# process today, so they run as a ctest entry of their own.
fuzzybits_add_imported_test(
  TARGET fzb_nombrilo_bug_tests
  SOURCES nombrilo_bug_tests.cc
  LINK_LIBRARIES fuzzybits_compile
  LABELS bug
  ARGS "~[crash]"
)
if(NOT APPLE AND NOT FUZZYBITS_WITH_MC)
  target_sources(fzb_nombrilo_bug_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/compile/direct_eval_host_stubs.cc")
  target_include_directories(fzb_nombrilo_bug_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/mc")
elseif(FUZZYBITS_WITH_MC)
  target_link_libraries(fzb_nombrilo_bug_tests PRIVATE mc)
endif()
add_test(NAME fzb_nombrilo_bug_tests.crash COMMAND fzb_nombrilo_bug_tests "[crash]")
set_tests_properties(fzb_nombrilo_bug_tests.crash PROPERTIES LABELS bug)

# Known bugs of fzb compile on OpenPSA MEF input (ctest -L bug): these fail until their issues are fixed. The
# test runs the fzb binary and links no SCRAM code, so it holds for the new OpenPSA reader too (#148).
if(NOT SKBUILD)
  fuzzybits_add_imported_test(
    TARGET fzb_mef_bug_tests
    SOURCES mef_bug_tests.cc
    LINK_LIBRARIES fuzzybits_exact_eval pcfb
    LABELS bug
  )
  target_compile_definitions(fzb_mef_bug_tests PRIVATE FZB_PATH="$<TARGET_FILE:fzb>")
  add_dependencies(fzb_mef_bug_tests fzb)
  set_tests_properties(fzb_mef_bug_tests PROPERTIES TIMEOUT 300)
endif()
