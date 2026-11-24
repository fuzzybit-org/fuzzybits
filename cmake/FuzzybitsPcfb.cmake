# pcfb (public, fuzzybit-org/pcfb) is a git submodule at extern/pcfb. Its gitlink
# is the only pin: bump it with `git -C extern/pcfb checkout <rev>` and commit the
# superproject. Sources, in order:
#   1. -DFUZZYBITS_PCFB_SOURCE_DIR=/path/to/pcfb (a local checkout; overrides the
#      submodule).
#   2. extern/pcfb, when the submodule is initialised
#      (`git submodule update --init extern/pcfb`).
#   3. FetchContent of the gitlink's own revision (read with git), for a source
#      tree whose submodule is not initialised.
# On Windows the pcfb sources are copied into the build tree and their POSIX mmap
# I/O is patched there (cmake/patches/pcfb_windows_mmap.cmake), so the submodule
# itself is never modified.

set(FUZZYBITS_PCFB_SOURCE_DIR "" CACHE PATH
    "Path to a local pcfb checkout (overrides the extern/pcfb submodule)")
set(FUZZYBITS_PCFB_GIT_REPO "https://github.com/fuzzybit-org/pcfb.git" CACHE STRING
    "pcfb git remote for the FetchContent fallback")

set(_fuzzybits_pcfb_submodule "${CMAKE_CURRENT_LIST_DIR}/../extern/pcfb")
cmake_path(NORMAL_PATH _fuzzybits_pcfb_submodule)

# The revision the superproject's gitlink records for extern/pcfb, or "" when git
# cannot tell (no .git, no git executable).
function(fuzzybits_pcfb_gitlink out_var)
  set(_rev "")
  find_package(Git QUIET)
  if(GIT_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse "HEAD:extern/pcfb"
      WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
      OUTPUT_VARIABLE _rev
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _status)
    if(NOT _status EQUAL 0)
      set(_rev "")
    endif()
  endif()
  set(${out_var} "${_rev}" PARENT_SCOPE)
endfunction()

function(fuzzybits_add_pcfb)
  if(TARGET pcfb)
    return()
  endif()

  # pcfb's CMakeLists enables its own Catch2 suite on BUILD_TESTING. Keep that
  # off while we import it — fuzzybits tests live under src/compile/tests.
  # include(CTest) inside pcfb would otherwise reset the cache option to ON.
  set(_fuzzybits_saved_testing "${BUILD_TESTING}")
  set(BUILD_TESTING OFF CACHE BOOL "Build the testing tree" FORCE)
  set(PCFB_BUILD_TESTS OFF CACHE BOOL "Build pcfb tests" FORCE)

  set(_source "")
  if(FUZZYBITS_PCFB_SOURCE_DIR)
    if(NOT EXISTS "${FUZZYBITS_PCFB_SOURCE_DIR}/CMakeLists.txt")
      message(FATAL_ERROR
        "FUZZYBITS_PCFB_SOURCE_DIR=${FUZZYBITS_PCFB_SOURCE_DIR} does not contain CMakeLists.txt")
    endif()
    set(_source "${FUZZYBITS_PCFB_SOURCE_DIR}")
    message(STATUS "pcfb: ${_source} (FUZZYBITS_PCFB_SOURCE_DIR)")
  elseif(EXISTS "${_fuzzybits_pcfb_submodule}/CMakeLists.txt")
    set(_source "${_fuzzybits_pcfb_submodule}")
    message(STATUS "pcfb: ${_source} (submodule)")
  else()
    fuzzybits_pcfb_gitlink(_rev)
    if(_rev STREQUAL "")
      message(FATAL_ERROR
        "pcfb: extern/pcfb is not initialised and its revision cannot be read from git. "
        "Run `git submodule update --init extern/pcfb`, or pass -DFUZZYBITS_PCFB_SOURCE_DIR=/path/to/pcfb.")
    endif()
    message(STATUS "pcfb: FetchContent ${FUZZYBITS_PCFB_GIT_REPO} @ ${_rev} (submodule not initialised)")
    include(FetchContent)
    FetchContent_Declare(pcfb
      GIT_REPOSITORY ${FUZZYBITS_PCFB_GIT_REPO}
      GIT_TAG        ${_rev}
    )
    FetchContent_Populate(pcfb)
    set(_source "${pcfb_SOURCE_DIR}")
  endif()

  if(WIN32)
    # Patch a copy, never the submodule or a user checkout.
    set(_copy "${CMAKE_BINARY_DIR}/_deps/pcfb-patched")
    file(REMOVE_RECURSE "${_copy}")
    file(COPY "${_source}/" DESTINATION "${_copy}" PATTERN ".git" EXCLUDE)
    include("${CMAKE_CURRENT_LIST_DIR}/patches/pcfb_windows_mmap.cmake")
    fuzzybits_pcfb_apply_windows_mmap("${_copy}")
    set(_source "${_copy}")
  endif()
  add_subdirectory("${_source}" "${CMAKE_BINARY_DIR}/_deps/pcfb-build" EXCLUDE_FROM_ALL)

  set(BUILD_TESTING "${_fuzzybits_saved_testing}" CACHE BOOL "Build the testing tree" FORCE)

  if(NOT TARGET pcfb)
    message(FATAL_ERROR "pcfb target was not created")
  endif()
  if(NOT TARGET pcfb::pcfb)
    add_library(pcfb::pcfb ALIAS pcfb)
  endif()
endfunction()

fuzzybits_add_pcfb()
unset(_fuzzybits_pcfb_submodule)
