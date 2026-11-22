add_library(fuzzybits_ir_pc INTERFACE)
target_link_libraries(fuzzybits_ir_pc INTERFACE fuzzybits_ir)

add_subdirectory(hull)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
