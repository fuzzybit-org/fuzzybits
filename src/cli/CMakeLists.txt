# Terminal progress renderer for the CLI front ends.
#
# Host-only (no SYCL, no third-party deps) and separate from the engine: it is consumer code,
# linked into the fzb CLI and the Python extension, never into libmc — so nothing in the engine
# links terminal code. It lives outside src/runtime because `fzb compile` builds without
# FUZZYBITS_WITH_MC and still needs the renderer.
add_library(fuzzybits_progress_line STATIC
  progress_line.cc
)
set_target_properties(fuzzybits_progress_line PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(fuzzybits_progress_line
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
