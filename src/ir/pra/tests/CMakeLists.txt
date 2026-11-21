fuzzybits_ensure_catch2()

fuzzybits_add_imported_test(
  TARGET fuzzybits_ir_pra_tests
  SOURCES model_tests.cc validation_tests.cc scale_tests.cc
  LINK_LIBRARIES fuzzybits_ir_pra fuzzybits_ir_pc
  LABELS unit
)
