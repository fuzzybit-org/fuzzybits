add_library(fuzzybits_ir_pra STATIC
  builder.cpp
  builder.h
  descriptor.h
  event.h
  expression.h
  fault_tree.h
  gate.h
  id.h
  metadata.h
  model.h
  pra.h
  target.h
)
set_target_properties(fuzzybits_ir_pra PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fuzzybits_ir_pra PUBLIC fuzzybits_ir)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
