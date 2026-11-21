# ir::pc -> ir::pc passes (#177). SCRAM-free: links only the IR and API types.
add_library(fuzzybits_ir_pc_passes STATIC
  coalesce.cpp
  eliminate.cpp
  fold.cpp
  pipeline.cpp
  propagate.cpp
  detail/rewrite.cpp
  detail/rewrite.h
  passes.h
  transform.h
)
set_target_properties(fuzzybits_ir_pc_passes PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fuzzybits_ir_pc_passes PUBLIC fuzzybits_ir_pc)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
