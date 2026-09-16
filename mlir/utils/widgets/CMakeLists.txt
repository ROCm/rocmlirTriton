# The hsaco widgets convert between an hsaco ELF and the escaped text rocmlirTriton
# embeds it as. They have to sit in ${ROCMLIR_BIN_DIR} because lit tests invoke
# them from PATH alongside rocmlir-opt (see
# mlir/test/Dialect/Rock/triton-to-hsaco-dump-invariant.mlir).
set(WIDGET_SCRIPTS
  hsacoDisasm.py
  hsacoToStringAttr.py
)

list(TRANSFORM WIDGET_SCRIPTS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

add_custom_target(rocmlir-widgets
  COMMAND ${CMAKE_COMMAND} -E copy ${WIDGET_SCRIPTS} ${ROCMLIR_BIN_DIR})
