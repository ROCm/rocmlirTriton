add_rocmlir_dialect_library(MLIRMIGraphXTransforms
  AnnotateFusions.cpp
  MIGraphXTransform.cpp
  RealizeInt4.cpp
  MIGraphXTosaSimplify.cpp

  DEPENDS
  MLIRMIGraphXPassIncGen

  LINK_LIBS PUBLIC
  MLIRMIGraphXDialect
  MLIRPass
  MLIRFuncDialect
  MLIRRockOps
  MLIRTosaDialect
  MLIRTransforms
  MLIRSupport
)
