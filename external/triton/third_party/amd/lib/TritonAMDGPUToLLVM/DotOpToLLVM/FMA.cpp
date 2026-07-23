#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace ::mlir::triton::gpu;

namespace {

struct DotIntrinsic {
  int vectorSize;
  Type outElemTy;
  StringRef intrinsicName;
  SmallVector<Value> additionalArgs;
};

class AMDFMAVectorMultiplier : public FMAVectorMultiplier {
  ConversionPatternRewriter &rewriter;
  Location loc;
  DotIntrinsic intrinsic;

  DotIntrinsic chooseIntrinsic(DotOp op) {
    auto aOpTy = cast<RankedTensorType>(op.getA().getType());
    auto aElemTy = aOpTy.getElementType();
    auto bElemTy = aOpTy.getElementType();
    assert(aElemTy == bElemTy);
    auto dOpTy = cast<RankedTensorType>(op.getD().getType());
    auto dElemTy = dOpTy.getElementType();
    DotIntrinsic chosenOp;

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    if ((aElemTy.isF16() || aElemTy.isBF16()) && dElemTy.isF32()) {
      chosenOp.vectorSize = 2;
      chosenOp.outElemTy = f32_ty;
      chosenOp.intrinsicName =
          aElemTy.isF16() ? "llvm.amdgcn.fdot2" : "llvm.amdgcn.fdot2.f32.bf16";
      chosenOp.additionalArgs = {b.false_val()};
      return chosenOp;
    }
    if (aElemTy.isInteger(8) && dElemTy.isInteger(32)) {
      chosenOp.vectorSize = 4;
      chosenOp.outElemTy = i32_ty;
      chosenOp.intrinsicName = "llvm.amdgcn.sdot4";
      chosenOp.additionalArgs = {b.false_val()};
      return chosenOp;
    }
    // TODO(gfx1170): emit v_dot4_f32_fp8/bf8 for fp8 blocked
    // (FMA) dots. gfx1170 and RDNA4 (gfx1200/1201) have the Dot11Insts fp8 dot4
    // ops (llvm.amdgcn.dot4.f32.{fp8,bf8}.{fp8,bf8}); the assembler accepts
    // them on those targets only (not RDNA3/gfx1250/CDNA). Wiring them here
    // would let non-WMMA fp8 dots use a K=4 packed dot instead of scalar
    // upcast+fmuladd:
    //   - add a supportsFp8Dot4() capability gated on {GFX1170, RDNA4} and read
    //     TargetFeatures::fromModuleOp() here to guard the branch;
    //   - for same-type operands map Float8E4M3FN->"fp8" / Float8E5M2->"bf8",
    //     vectorSize=4, outElemTy=f32, and extend packOperand() to bitcast the
    //     4xfp8 vector to i32 (as done for i8);
    //   - the mixed variants (fp8_bf8 / bf8_fp8) additionally need the shared
    //     FMADotUtility multiplier to carry separate A/B element types.
    // Currently unreachable: fp8 GEMM on gfx1170 goes through WMMA v2, and the
    // blocked/FMA fallback upcasts fp8 before the dot, so this is a latent gap
    // rather than an active miscompile.
    // choose one of FMA intrinsics
    assert(aElemTy.isIntOrFloat() && !aElemTy.isIntOrIndex());
    assert(aElemTy == dElemTy);
    assert(cast<RankedTensorType>(op.getA().getType()).getElementType() ==
           dElemTy);
    chosenOp.vectorSize = 1;
    chosenOp.outElemTy = aElemTy;
    if (aElemTy.isF64())
      chosenOp.intrinsicName = "llvm.fmuladd.f64";
    if (aElemTy.isF32())
      chosenOp.intrinsicName = "llvm.fmuladd.f32";
    if (aElemTy.isF16())
      chosenOp.intrinsicName = "llvm.fmuladd.f16";
    chosenOp.additionalArgs = {};
    return chosenOp;
  }

  Value packOperand(ArrayRef<Value> scalarValues, int firstElemPos,
                    unsigned vectorSize) {
    if (vectorSize == 1)
      return scalarValues[firstElemPos];
    auto elemTy = scalarValues[firstElemPos].getType();
    auto vecTy = vec_ty(elemTy, vectorSize);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value vec = b.undef(vecTy);
    for (int elem = 0; elem < vectorSize; ++elem) {
      int elemPos = firstElemPos + elem;
      vec =
          b.insert_element(vecTy, vec, scalarValues[elemPos], b.i32_val(elem));
    }
    if (elemTy.isInteger(8)) {
      assert(vectorSize == 4);
      vec = b.bitcast(vec, i32_ty);
    }
    return vec;
  }

  Value generateDotInstr(Value a, Value b, Value c) {
    SmallVector<Value> args{a, b, c};
    args.append(intrinsic.additionalArgs.begin(),
                intrinsic.additionalArgs.end());
    SmallVector<Type> argTypes;
    for (auto arg : args)
      argTypes.push_back(arg.getType());
    auto d = LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, intrinsic.intrinsicName, intrinsic.outElemTy, args);
    return d.getResult(0);
  }

public:
  AMDFMAVectorMultiplier(ConversionPatternRewriter &rewriter, DotOp op)
      : rewriter(rewriter), loc(op.getLoc()), intrinsic(chooseIntrinsic(op)) {}

  Value multiplyVectors(ArrayRef<Value> a, ArrayRef<Value> b,
                        Value c) override {
    auto kSize = a.size();
    assert(b.size() == kSize);
    Value accum = c;
    for (int k = 0; k < kSize; k += intrinsic.vectorSize) {
      auto aOp = packOperand(a, k, intrinsic.vectorSize);
      auto bOp = packOperand(b, k, intrinsic.vectorSize);
      accum = generateDotInstr(aOp, bOp, accum);
    }
    return accum;
  }
};

} // namespace

namespace mlir::triton::AMD {

LogicalResult convertAMDFMADot(DotOp op, DotOp::Adaptor adaptor,
                               const LLVMTypeConverter *typeConverter,
                               ConversionPatternRewriter &rewriter) {
  AMDFMAVectorMultiplier multiplier(rewriter, op);
  return parametricConvertFMADot(op, adaptor, typeConverter, rewriter,
                                 multiplier);
}
} // namespace mlir::triton::AMD
