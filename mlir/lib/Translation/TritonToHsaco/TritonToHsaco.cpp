//===- TritonToHsaco.cpp - Convert Triton LLVM IR to HSACO binary --------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides:
// 1. A translation that converts Triton LLVM dialect IR to HSACO binary format
// 2. A pass wrapper that can be used in pipelines
//
// It implements the functionality from Triton's make_llir(), make_amdgcn(),
// and make_hsaco() in compiler.py.
//
//===----------------------------------------------------------------------===//

#include "mlir/Translation/TritonToHsaco.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/KnobUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Config/Targets.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"

#include <array>
#include <mutex>
#include <unordered_set>

// LLD for linking
#if LLVM_HAS_AMDGPU_TARGET
#include "lld/Common/Driver.h"
LLD_HAS_DRIVER(elf)
#endif

#define DEBUG_TYPE "triton-to-hsaco"

// Forward declaration for Triton's BreakStructPhiNodesPass (same as llvm.cc)
// Implementation is in lib/Target/LLVMIR/LLVMIRBreakPhiStruct.cpp
namespace llvm {
struct BreakStructPhiNodesPass : PassInfoMixin<BreakStructPhiNodesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "BreakStructPhiNodesPass"; }
};
} // namespace llvm

// Forward declaration for Triton's scalarize pass
namespace mlir::triton::AMD {
void runScalarizePackedFOpsPass(llvm::Function &F);
}

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Diagnostic handler that swallows one specific LLVM optimization-failure
/// warning emitted by the AMDGPU backend.
///
/// The default `llvm::DiagnosticHandler` (installed by `LLVMContextImpl`)
/// is a no-op stub whose `handleDiagnostics` returns `false`, causing
/// `LLVMContext::diagnose` to fall through to its built-in stderr printer.
/// For a `DS_Error` diagnostic, `diagnose` records `HasErrors` on the
/// handler (it no longer aborts the process); the caller is responsible for
/// inspecting that flag and failing. Returning `false` here preserves that
/// built-in behaviour for every diagnostic that doesn't match the predicate
class SuppressWarningHandler : public llvm::DiagnosticHandler {
public:
  bool handleDiagnostics(const llvm::DiagnosticInfo &diag) override {
    auto *optDiag = llvm::dyn_cast<llvm::DiagnosticInfoOptimizationBase>(&diag);
    if (!optDiag)
      return false;

    std::string msg = optDiag->getMsg();
    if (diag.getKind() == llvm::DK_OptimizationFailure &&
        diag.getSeverity() == llvm::DS_Warning &&
        llvm::StringRef(msg).starts_with(
            "failed to meet occupancy target given by 'amdgpu-waves-per-eu'"))
      return true;

    return false;
  }
};

/// Initialize LLVM targets (call once) - from init_targets in llvm.cc
void initializeLLVMTargets() {
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
  });
  // Disable LLVM's internal parallelism. Triton kernels produce small LLVM
  // modules where pass-level parallelism is not beneficial, and LLVM's
  // global thread pool is not fork-safe: a forked child inherits the pool's
  // state but not its threads, causing SIGABRT on use or cleanup.
  llvm::parallel::strategy = llvm::hardware_concurrency(1);
}

/// Create LLVM target machine - from createTargetMachine in llvm.cc
std::unique_ptr<llvm::TargetMachine> createTargetMachine(llvm::Module &module,
                                                         llvm::Triple &triple,
                                                         StringRef archStr,
                                                         StringRef features,
                                                         bool enableFpFusion) {
  std::string error;
  auto *target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) {
    llvm::errs() << "Target lookup failed: " << error << "\n";
    return nullptr;
  }

  llvm::TargetOptions opt;
  if (enableFpFusion)
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.TrapUnreachable = true;
  opt.MCOptions.AsmVerbose = true;
  opt.MCOptions.PreserveAsmComments = true;

  return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      triple, archStr, features, opt, llvm::Reloc::PIC_, std::nullopt,
      llvm::CodeGenOptLevel::Aggressive));
}

/// Add control constant to module (for device library compatibility)
void addControlConstant(llvm::Module &module, const char *name, int bitwidth,
                        int value) {
  llvm::Type *type =
      llvm::IntegerType::getIntNTy(module.getContext(), bitwidth);
  auto *gv = new llvm::GlobalVariable(
      module, type, /*isConstant=*/true,
      llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
      llvm::ConstantInt::get(type, value), name, nullptr,
      llvm::GlobalValue::ThreadLocalMode::NotThreadLocal, /*AddressSpace=*/4);
  gv->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);
  gv->setVisibility(llvm::GlobalValue::VisibilityTypes::ProtectedVisibility);
  gv->setAlignment(llvm::MaybeAlign(bitwidth / 8));
}

/// Set ISA version control constant
void setISAVersion(llvm::Module &module, StringRef archStr) {
  llvm::AMDGPU::IsaVersion version = llvm::AMDGPU::getIsaVersion(archStr);
  int isaVersion =
      version.Major * 1000 + version.Minor * 100 + version.Stepping;
  addControlConstant(module, "__oclc_ISA_version", /*bitwidth=*/32, isaVersion);
}

/// Set ABI version control constant
void setABIVersion(llvm::Module &module, int version) {
  llvm::Type *i32Ty = llvm::Type::getInt32Ty(module.getContext());
  auto *gv = new llvm::GlobalVariable(
      module, i32Ty, /*isConstant=*/true,
      llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
      llvm::ConstantInt::get(i32Ty, version), "__oclc_ABI_version", nullptr,
      llvm::GlobalValue::ThreadLocalMode::NotThreadLocal, /*AddressSpace=*/4);
  gv->setVisibility(llvm::GlobalValue::VisibilityTypes::ProtectedVisibility);
  gv->setAlignment(llvm::MaybeAlign(4));
  gv->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);

  module.addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version",
                       version);
}

/// Mirrors upstream compiler.py `is_coexec_scheduler_supported(arch)`.
static bool isCoexecSchedulerSupported(llvm::StringRef arch) {
  return arch.starts_with("gfx1250");
}

/// Set kernel function attributes
void setKernelAttributes(llvm::Module &module, StringRef archStr,
                         StringRef features, int numWarps, int wavesPerEU,
                         int numCTAs, bool allowFlushDenorm, bool enableAsan,
                         bool enableExpertScheduling, StringRef llvmFnAttrs) {
  int waveSize = rock::getWaveSize(archStr);
  int totalThreads = numWarps * waveSize;

  // Match compiler.py: the kernel is the only non-declaration function with
  // external linkage; instrumentation helpers (e.g. ConSan) use internal
  // linkage. If none is found, we cannot produce a valid HSACO.
  llvm::Function *kernelFn = nullptr;
  for (llvm::Function &fn : module) {
    if (!fn.isDeclaration() && fn.hasExternalLinkage()) {
      kernelFn = &fn;
      break;
    }
  }

  kernelFn->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
  kernelFn->addFnAttr("amdgpu-cluster-dims", std::to_string(numCTAs) + ",1,1");
  kernelFn->addFnAttr("amdgpu-flat-work-group-size",
                      "1," + std::to_string(totalThreads));

  kernelFn->addFnAttr("uniform-work-group-size", "true");

  if (wavesPerEU > 0) {
    std::string wavesStr =
        std::to_string(wavesPerEU) + ", " + std::to_string(wavesPerEU);
    kernelFn->addFnAttr("amdgpu-waves-per-eu", wavesStr);
  }

  // gfx1250 coexec scheduler hint. Mirrors upstream compiler.py make_llir():
  //   if is_coexec_scheduler_supported(options.arch) and options.num_warps <=
  //   4:
  //       kernel_fn.add_fn_attr("amdgpu-sched-strategy", "coexec")
  // Added after waves-per-eu, matching upstream order.
  if (isCoexecSchedulerSupported(archStr) && numWarps <= 4) {
    kernelFn->addFnAttr("amdgpu-sched-strategy", "coexec");
  }

  // Deliberate divergence from upstream Triton: compiler.py passes
  // "amdgpu-expert-scheduling-mode" as a translate_to_asm flag, which the
  // Python llvm.cc binding applies by mutating LLVM's process-global cl::opt.
  // rocmlir-tuning-driver compiles configs concurrently in one process, so
  // stamp the backend's per-function attribute on every defined function
  // instead. This keeps upstream's "all functions" behavior without touching
  // process-global state. SIInsertWaitcnts reads this attribute when the global
  // option was not set on the process command line.
  for (llvm::Function &fn : module) {
    if (!fn.isDeclaration())
      fn.addFnAttr("amdgpu-expert-scheduling-mode",
                   enableExpertScheduling ? "true" : "false");
  }

  std::string denormalMode = allowFlushDenorm ? "preserve-sign" : "ieee";
  kernelFn->addFnAttr("denormal-fp-math-f32", denormalMode);

  // ASan support
  // Only stamp `target-features` on the kernel when the caller actually has
  // something to override with (e.g. `+xnack` for asan). Stamping an empty
  // override causes LLVM's per-function subtarget lookup to key off a bare
  // `"target-features"=""` attribute and silently *ignore* the TM-level
  // `-mattr` we later set on `tmAsm` in `make_amdgcn` (which is where
  // `-real-true16` is added for gfx11 kernels). Upstream Triton only
  // touches `target-features` in the asan path (see
  // `add_fn_target_feature("+xnack")` in compiler.py).
  if (enableAsan) {
    kernelFn->addFnAttr("target-features", features);
    kernelFn->addFnAttr(llvm::Attribute::SanitizeAddress);
  }

  // Debug-only developer overrides, applied last so they win over the
  // attributes stamped above. Mirrors the experimental `llvm_fn_attrs` loop in
  // compiler.py make_llir():
  //   for name, value in options.llvm_fn_attrs:
  //       kernel_fn.remove_fn_attr(name)
  //       kernel_fn.add_fn_attr(name, value)
  // A bare `name` produces a valueless attribute; `name=value` sets a value
  // (an empty value, e.g. `name=`, is also treated as valueless).
  if (!llvmFnAttrs.empty()) {
    llvm::SmallVector<StringRef> attrs;
    llvmFnAttrs.split(attrs, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (StringRef attr : attrs) {
      std::pair<StringRef, StringRef> kv = attr.split('=');
      StringRef name = kv.first.trim();
      if (name.empty())
        continue;
      StringRef value = kv.second.trim();
      kernelFn->removeFnAttr(name);
      if (value.empty())
        kernelFn->addFnAttr(name);
      else
        kernelFn->addFnAttr(name, value);
    }
  }

  // set_all_fn_arg_inreg in compiler.py
  if (!archStr.starts_with("gfx1250")) {
    for (llvm::Argument &arg : kernelFn->args()) {
      if (!arg.hasByRefAttr() && !arg.hasNestAttr()) {
        arg.addAttr(llvm::Attribute::InReg);
      }
    }
  }
}

/// Check if architecture has architected SGPRs
bool hasArchitectedSGPRs(llvm::Triple &triple, StringRef archStr) {
  std::string error;
  auto *target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target)
    return false;

  std::unique_ptr<llvm::MCSubtargetInfo> sti(
      target->createMCSubtargetInfo(triple, archStr, ""));
  return sti && sti->checkFeatures("+architected-sgprs");
}

/// Link external device libraries (ocml, ockl, etc.)
bool linkExternLibs(llvm::Module &module,
                    const std::vector<std::string> &paths) {
  if (paths.empty())
    return true;

  llvm::LLVMContext &ctx = module.getContext();
  llvm::Linker linker(module);

  for (const std::string &path : paths) {
    llvm::SMDiagnostic err;
    std::unique_ptr<llvm::Module> libMod = llvm::parseIRFile(path, err, ctx);
    if (!libMod) {
      llvm::errs() << "Failed to parse library at " << path << "\n";
      return false;
    }
    libMod->setTargetTriple(llvm::Triple(module.getTargetTriple()));
    libMod->setDataLayout(module.getDataLayout());

    std::unordered_set<std::string> externalFns;
    for (llvm::Function &fn : libMod->functions()) {
      if (!fn.isDeclaration())
        externalFns.insert(fn.getName().str());
    }

    if (linker.linkInModule(std::move(libMod),
                            llvm::Linker::Flags::LinkOnlyNeeded)) {
      llvm::errs() << "Failed to link library at " << path << "\n";
      return false;
    }

    // Mark linked-in functions as internal
    for (llvm::Function &fn : module.functions()) {
      if (externalFns.count(fn.getName().str())) {
        fn.setLinkage(llvm::GlobalValue::InternalLinkage);
      }
    }
  }
  return true;
}

static std::optional<llvm::OptimizationLevel> mapToLevel(unsigned optLevel) {
  switch (optLevel) {
  case 0:
    return llvm::OptimizationLevel::O0;
  case 1:
    return llvm::OptimizationLevel::O1;
  case 2:
    return llvm::OptimizationLevel::O2;
  case 3:
    return llvm::OptimizationLevel::O3;
  }
  return std::nullopt;
}

/// Run LLVM optimization passes - matches optimize_module in llvm.cc
void optimizeModule(llvm::Module &module, llvm::TargetMachine *tm,
                    StringRef arch, llvm::OptimizationLevel optLevel,
                    bool enableAsan) {
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  llvm::PipelineTuningOptions tuningOptions;
  tuningOptions.LoopUnrolling = true;
  tuningOptions.LoopInterleaving = true;
  tuningOptions.LoopVectorization = true;
  tuningOptions.SLPVectorization = true;

  // Disable the VectorCombine pass. Mirrors upstream compiler.py make_llir(),
  // which now calls `llvm.optimize_module(..., disable_vector_combine=True)`;
  // in llvm.cc that registers a should-run callback skipping VectorCombinePass.
  // VectorCombinePass::name() returns the C++ class name, not the registry
  // name "vector-combine".
  llvm::PassInstrumentationCallbacks pic;
  pic.registerShouldRunOptionalPassCallback(
      [](llvm::StringRef passName, llvm::Any) {
        return passName != "VectorCombinePass";
      });

  llvm::PassBuilder pb(tm, tuningOptions, /*PGOOpt=*/std::nullopt, &pic);

  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::ModulePassManager mpm;

  // Register callback to add BreakStructPhiNodesPass before vectorization
  // This matches llvm.cc's registerVectorizerStartEPCallback
  pb.registerVectorizerStartEPCallback(
      [&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
        // Triton generates large structure of scalars which may pessimise
        // optimizations, we run a pass to break up phi of struct to make
        // sure all the struct are removed for the following passes.
        fpm.addPass(llvm::BreakStructPhiNodesPass());
        fpm.addPass(llvm::InstCombinePass());
      });

  // Add address sanitizer if enabled
  if (enableAsan) {
    llvm::AddressSanitizerOptions asanOpts;
    mpm.addPass(llvm::AddressSanitizerPass(asanOpts));
  }

  mpm.addPass(pb.buildPerModuleDefaultPipeline(optLevel));
  mpm.run(module, mam);
}

/// Clean up metadata (cleanup_bitcode_metadata in compiler.py)
void cleanupBitcodeMetadata(llvm::Module &module) {
  if (auto *ident = module.getNamedMetadata("llvm.ident"))
    module.eraseNamedMetadata(ident);
  if (auto *openclVersion = module.getNamedMetadata("opencl.ocl.version"))
    module.eraseNamedMetadata(openclVersion);
}

/// Disable inlining of print related functions (disable_print_inline)
void disablePrintInline(llvm::Module &module) {
  // List of functions name prefixes we want to forbid inline.
  std::array<const char *, 2> prefixes = {"__ockl_fprintf", "__ockl_printf"};

  for (llvm::Function &f : module) {
    if (!f.hasName())
      continue;
    llvm::StringRef name = f.getName();

    auto isNamePrefixed = [&name](const char *prefix) {
      return name.starts_with(prefix);
    };

    if (llvm::any_of(prefixes, isNamePrefixed))
      f.addFnAttr(llvm::Attribute::NoInline);
  }
}

//===----------------------------------------------------------------------===//
// Register-pressure gate
//===----------------------------------------------------------------------===//

/// Registers a value of `ty` occupies. `getRegUsageForType` is the target's own
/// answer (the same hook the loop vectorizer weights its register-pressure
/// estimate with), but it only accepts types the target legalizes, so
/// aggregates and other leftovers fall back to their 32-bit lane count. Types
/// that never live in a register contribute nothing.
static uint64_t registerLanes(llvm::Type *ty, const llvm::DataLayout &dl,
                              const llvm::TargetTransformInfo &tti) {
  if (ty->isVoidTy() || ty->isLabelTy() || ty->isMetadataTy() ||
      ty->isTokenTy() || !ty->isSized())
    return 0;
  llvm::TypeSize bits = dl.getTypeSizeInBits(ty);
  if (bits.isScalable())
    return 0;
  if (ty->isFirstClassType() && !ty->isAggregateType())
    return tti.getRegUsageForType(ty);
  return llvm::divideCeil(bits.getFixedValue(), 32);
}

/// Only instruction results and arguments occupy a register across a program
/// point; constants and globals are rematerialized where needed.
static bool occupiesRegister(const llvm::Value *v) {
  return llvm::isa<llvm::Instruction>(v) || llvm::isa<llvm::Argument>(v);
}

/// Peak register pressure in `fn`: the largest total register width of
/// simultaneously-live SSA values at any point in the function.
///
/// LLVM has no reusable IR-level version of this. The loop vectorizer's
/// `calculateRegisterUsageForPlan` computes the same quantity but only over a
/// VPlan, and the AMDGPU backend's `GCNRegPressure` needs `LiveIntervals`, so
/// it is only available inside the codegen this check exists to avoid entering.
///
/// Standard backward liveness over the CFG (phi operands are attributed to the
/// incoming edge's block, which is where they must be available), followed by a
/// backward scan of each block that tracks the running live width. Linear in
/// the function per dataflow round.
static uint64_t peakRegisterPressure(const llvm::Function &fn,
                                     const llvm::TargetTransformInfo &tti) {
  using ValueSet = llvm::SmallPtrSet<const llvm::Value *, 32>;
  const llvm::DataLayout &dl = fn.getParent()->getDataLayout();

  // Per-block `use` (read before written in the block) and `def` sets.
  llvm::DenseMap<const llvm::BasicBlock *, ValueSet> uses, defs;
  for (const llvm::BasicBlock &bb : fn) {
    ValueSet &use = uses[&bb];
    ValueSet &def = defs[&bb];
    for (const llvm::Instruction &inst : bb) {
      if (!llvm::isa<llvm::PHINode>(&inst))
        for (const llvm::Value *op : inst.operand_values())
          if (occupiesRegister(op) && !def.contains(op))
            use.insert(op);
      if (occupiesRegister(&inst))
        def.insert(&inst);
    }
  }

  llvm::DenseMap<const llvm::BasicBlock *, ValueSet> liveIn, liveOut;
  bool changed = true;
  while (changed) {
    changed = false;
    for (const llvm::BasicBlock &bb : llvm::reverse(fn)) {
      ValueSet out;
      for (const llvm::BasicBlock *succ : llvm::successors(&bb)) {
        out.insert_range(liveIn[succ]);
        // A phi's incoming value must be live out of the edge it arrives on.
        for (const llvm::PHINode &phi : succ->phis()) {
          const llvm::Value *in = phi.getIncomingValueForBlock(&bb);
          if (in && occupiesRegister(in))
            out.insert(in);
        }
      }
      ValueSet in = uses[&bb];
      for (const llvm::Value *v : out)
        if (!defs[&bb].contains(v))
          in.insert(v);
      if (in.size() != liveIn[&bb].size() ||
          out.size() != liveOut[&bb].size()) {
        changed = true;
        liveIn[&bb] = std::move(in);
        liveOut[&bb] = std::move(out);
      }
    }
  }

  uint64_t peak = 0;
  for (const llvm::BasicBlock &bb : fn) {
    ValueSet live = liveOut[&bb];
    uint64_t width = 0;
    for (const llvm::Value *v : live)
      width += registerLanes(v->getType(), dl, tti);
    peak = std::max(peak, width);
    for (const llvm::Instruction &inst : llvm::reverse(bb)) {
      if (live.erase(&inst))
        width -= registerLanes(inst.getType(), dl, tti);
      // Phi operands belong to the predecessor blocks, not to this point.
      if (!llvm::isa<llvm::PHINode>(&inst))
        for (const llvm::Value *op : inst.operand_values())
          if (occupiesRegister(op) && live.insert(op).second)
            width += registerLanes(op->getType(), dl, tti);
      peak = std::max(peak, width);
    }
  }
  return peak;
}

/// Register width `fn` carries across its loop back edges: the total width of
/// its phi nodes.
///
/// A phi is a value the loop hands to its next iteration -- an accumulator, or
/// a buffer a pipelined loop has prefetched -- so unlike an ordinary result,
/// which dies a few instructions after it is defined, it is live across the
/// whole body by construction. That makes carried width the part of the
/// pressure the register allocator cannot relieve by splitting: past the
/// register file, every carried value spills and reloads once per iteration.
///
/// Peak pressure does not see this. Two configurations can hold the same amount
/// live at their worst point and differ by an order of magnitude in how much of
/// it they carry, and it is the carried part that decides whether allocation
/// converges or thrashes.
static uint64_t carriedRegisterLanes(const llvm::Function &fn,
                                     const llvm::TargetTransformInfo &tti) {
  const llvm::DataLayout &dl = fn.getParent()->getDataLayout();
  uint64_t lanes = 0;
  for (const llvm::BasicBlock &bb : fn)
    for (const llvm::PHINode &phi : bb.phis())
      lanes += registerLanes(phi.getType(), dl, tti);
  return lanes;
}

/// Refuse the configuration if any kernel in `llvmModule` asks for more
/// registers than a thread on `arch` can address: `peakPercent` bounds what may
/// be live at once and `carriedPercent` what may be carried across a loop back
/// edge, both as a percentage of the addressable file. A zero percentage skips
/// that bound.
///
/// Such a kernel spills in proportion, which both makes it several times slower
/// to run than a tile that fits and hands the register allocator a problem that
/// dominates compile time. Neither is worth waiting for, so mark the module
/// inapplicable and let tuning take the next config. `stage` names the point in
/// the pipeline for the diagnostic.
///
/// The two bounds answer different questions and neither subsumes the other.
/// Peak width is loose because most of what is live at the worst point can be
/// split or rematerialized, so it only bounds the total, and it takes a
/// multiple of the file before it means trouble. Carried width is the part that
/// cannot be relieved, so it is held to roughly the file itself.
///
/// Calibrated on a 1x512x8x8 f16 conv with a 512x512x3x3 filter, sweeping
/// kPerBlock over a 16x64 tile on one wave, which is the family that exposed
/// this.
static LogicalResult checkRegisterPressure(ModuleOp module,
                                           llvm::Module &llvmModule,
                                           llvm::TargetMachine *tm,
                                           StringRef arch, uint64_t peakPercent,
                                           uint64_t carriedPercent,
                                           StringRef stage) {
  const uint64_t addressable = rock::getAddressableVGPRs(arch);
  const uint64_t peakLimit = peakPercent * addressable / 100;
  const uint64_t carriedLimit = carriedPercent * addressable / 100;
  for (llvm::Function &fn : llvmModule) {
    if (fn.isDeclaration() || !fn.hasExternalLinkage())
      continue;
    llvm::TargetTransformInfo tti = tm->getTargetTransformInfo(fn);
    uint64_t peak = peakRegisterPressure(fn, tti);
    uint64_t carried = carriedRegisterLanes(fn, tti);
    LLVM_DEBUG(llvm::dbgs()
               << "[register-pressure] " << stage << " " << fn.getName() << ": "
               << peak << " lanes live (limit " << peakLimit << "), " << carried
               << " carried (limit " << carriedLimit << ")\n");
    const char *what = nullptr;
    uint64_t lanes = 0, limit = 0;
    if (peakPercent && peak > peakLimit) {
      what = "live at once";
      lanes = peak;
      limit = peakLimit;
    } else if (carriedPercent && carried > carriedLimit) {
      what = "carried across a loop back edge";
      lanes = carried;
      limit = carriedLimit;
    } else {
      continue;
    }
    rock::markAsNotApplicable(module);
    module.emitError() << "'" << fn.getName() << "' needs " << lanes
                       << " 32-bit register lanes " << what << " in the "
                       << stage << " IR, over the limit of " << limit << " for "
                       << arch
                       << "; the kernel would spill heavily and the register "
                          "allocator would dominate compile time";
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// make_amdgcn - LLVM IR to AMDGCN assembly (compiler.py lines 452-473)
// Inspired by translateLLVMIRToASM in external/triton/python/src/llvm.cc
//===----------------------------------------------------------------------===//

std::string translateLLVMIRToASM(llvm::Module &module,
                                 llvm::TargetMachine *machine) {
  using namespace mlir;

  // inline everything (matches llvm.cc lines 329-332)
  for (llvm::Function &f : module.functions())
    if (!f.hasFnAttribute(llvm::Attribute::NoInline))
      f.addFnAttr(llvm::Attribute::AlwaysInline);

  // verify and run inliner (matches llvm.cc lines 333-344)
  llvm::legacy::PassManager pm;
  pm.add(llvm::createAlwaysInlinerLegacyPass());
  pm.add(llvm::createVerifierPass());
  pm.run(module);

  // emit machine code (matches llvm.cc lines 360-377)
  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager pass;
    // emit
    machine->addPassesToEmitFile(pass, pstream, nullptr,
                                 llvm::CodeGenFileType::AssemblyFile);
    pass.run(module);
  }
  return result;
}

/// Translate LLVM IR module to AMDGCN assembly string
std::string makeAMDGCN(llvm::Module &module, llvm::TargetMachine *tm) {
  return translateLLVMIRToASM(module, tm);
}

//===----------------------------------------------------------------------===//
// make_hsaco - AMDGCN assembly to HSACO binary (compiler.py lines 476-488)
//===----------------------------------------------------------------------===//

/// Assemble AMDGCN assembly to object code (amd.assemble_amdgcn)
std::optional<SmallVector<char, 0>> assembleAMDGCN(StringRef assembly,
                                                   llvm::Triple &triple,
                                                   StringRef archStr,
                                                   StringRef features) {
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) {
    llvm::errs() << "Target lookup error: " << error << "\n";
    return std::nullopt;
  }

  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(assembly),
                            llvm::SMLoc());

  llvm::MCTargetOptions mcOptions;
  std::unique_ptr<llvm::MCRegisterInfo> mri(target->createMCRegInfo(triple));
  std::unique_ptr<llvm::MCAsmInfo> mai(
      target->createMCAsmInfo(*mri, triple, mcOptions));
  std::unique_ptr<llvm::MCSubtargetInfo> sti(
      target->createMCSubtargetInfo(triple, archStr, features));

  llvm::MCContext ctx(triple, *mai, *mri, *sti, &srcMgr);
  std::unique_ptr<llvm::MCObjectFileInfo> mofi(
      target->createMCObjectFileInfo(ctx, /*PIC=*/false,
                                     /*LargeCodeModel=*/false));
  ctx.setObjectFileInfo(mofi.get());

  llvm::SmallString<128> cwd;
  if (!llvm::sys::fs::current_path(cwd))
    ctx.setCompilationDir(cwd);

  llvm::SmallVector<char, 0> result;
  llvm::raw_svector_ostream svos(result);

  std::unique_ptr<llvm::MCInstrInfo> mcii(target->createMCInstrInfo());
  std::unique_ptr<llvm::MCCodeEmitter> ce(
      target->createMCCodeEmitter(*mcii, ctx));
  std::unique_ptr<llvm::MCAsmBackend> mab(
      target->createMCAsmBackend(*sti, *mri, mcOptions));
  std::unique_ptr<llvm::MCObjectWriter> ow(mab->createObjectWriter(svos));
  std::unique_ptr<llvm::MCStreamer> mcStreamer(target->createMCObjectStreamer(
      triple, ctx, std::move(mab), std::move(ow), std::move(ce), *sti));

  std::unique_ptr<llvm::MCAsmParser> parser(
      createMCAsmParser(srcMgr, ctx, *mcStreamer, *mai));
  std::unique_ptr<llvm::MCTargetAsmParser> tap(
      target->createMCAsmParser(*sti, *parser, *mcii));
  if (!tap) {
    llvm::errs() << "Assembler initialization error\n";
    return std::nullopt;
  }

  parser->setTargetParser(*tap);
  parser->Run(/*NoInitialTextSection=*/false);

  return SmallVector<char, 0>(result.begin(), result.end());
}

/// Invoke LLD to link object file to HSACO - matches triton_amd.cc lldInvoke
static std::optional<std::string> lldInvoke(const char *inPath,
                                            const char *outPath) {
#if LLVM_HAS_AMDGPU_TARGET
  // Workaround: Disable parallelism to avoid hangs caused by LLVM's thread pool
  // when the following code is executed in a forked child process.
  // Context: lld::elf::LinkerDriver::link uses parallelFor which uses the
  // LLVM's thread pool. During cleanup at ~TaskGroup() the child process hangs
  // waiting.
  //
  // IMPORTANT: LLD's CommonLinkerContext uses a global static pointer (not
  // thread_local) to store the linker context. This means lldMain is NOT
  // thread-safe - concurrent calls will race on the global context pointer.
  // We must serialize all LLD invocations with a mutex.
  static std::mutex lldMutex;
  std::lock_guard<std::mutex> lock(lldMutex);

  std::array<const char *, 6> args = {"ld.lld", "--threads=1", "-shared",
                                      inPath,   "-o",          outPath};
  std::string errString;
  llvm::raw_string_ostream errStream(errString);
  auto lldRes = lld::lldMain(args, llvm::outs(), errStream,
                             {{lld::Gnu, &lld::elf::link}});
  bool noErrors = (!lldRes.retCode && lldRes.canRunAgain);
  if (!noErrors) {
    errStream.flush();
    return errString;
  }
  return std::nullopt;
#else
  return "ROCM conversions not enabled";
#endif
}

/// Link object file to HSACO using LLD (amd.link_hsaco)
std::optional<SmallVector<char, 0>> linkHSACO(ArrayRef<char> objectCode) {
#if LLVM_HAS_AMDGPU_TARGET
  int tempObjFd = -1;
  llvm::SmallString<128> tempObjFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "o", tempObjFd,
                                         tempObjFilename)) {
    llvm::errs() << "Failed to create temporary object file\n";
    return std::nullopt;
  }
  llvm::FileRemover cleanupObj(tempObjFilename);
  {
    llvm::raw_fd_ostream tempObjOs(tempObjFd, true);
    tempObjOs << StringRef(objectCode.data(), objectCode.size());
    tempObjOs.flush();
  }

  llvm::SmallString<128> tempHsacoFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hsaco",
                                         tempHsacoFilename)) {
    llvm::errs() << "Failed to create temporary HSACO file\n";
    return std::nullopt;
  }
  llvm::FileRemover cleanupHsaco(tempHsacoFilename);

  // Use lldMain for safe re-entry support (matches triton_amd.cc)
  auto errOpt = lldInvoke(tempObjFilename.c_str(), tempHsacoFilename.c_str());
  if (errOpt) {
    llvm::errs() << "LLD invocation error: " << *errOpt << "\n";
    return std::nullopt;
  }

  auto hsacoFile =
      llvm::MemoryBuffer::getFile(tempHsacoFilename, /*IsText=*/false);
  if (!hsacoFile) {
    llvm::errs() << "Failed to read HSACO file\n";
    return std::nullopt;
  }

  StringRef buffer = (*hsacoFile)->getBuffer();
  return SmallVector<char, 0>(buffer.begin(), buffer.end());
#else
  llvm::errs() << "AMDGPU target not built. Rebuild LLVM with AMDGPU in "
                  "LLVM_TARGETS_TO_BUILD\n";
  return std::nullopt;
#endif
}

/// Convert AMDGCN assembly to HSACO binary - make_hsaco in compiler.py
std::optional<SmallVector<char, 0>> makeHSACO(StringRef amdgcnAsm,
                                              llvm::Triple &triple,
                                              StringRef archStr,
                                              StringRef features) {
  // Assemble to object code
  auto objectCode = assembleAMDGCN(amdgcnAsm, triple, archStr, features);
  if (!objectCode) {
    return std::nullopt;
  }

  // Link to HSACO
  return linkHSACO(*objectCode);
}

/// Mirrors upstream compiler.py `is_expert_scheduling_enabled(arch)`. gfx1250
/// is the only arch with AMDGPU expert-scheduling codegen support. Upstream
/// enables it there by default but honors the optional
/// `knobs.amd.use_expert_scheduling` (env `TRITON_HIP_USE_EXPERT_SCHEDULING`)
/// override
static bool isExpertSchedulingEnabled(llvm::StringRef arch,
                                      int useExpertScheduling) {
  if (!arch.starts_with("gfx1250"))
    return false;
  if (useExpertScheduling == mlir::rock::kKnobDefault)
    return true;
  return useExpertScheduling != 0;
}

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

namespace mlir {
namespace rock {

static void appendFeature(std::string &features, llvm::StringRef feature) {
  if (!features.empty())
    features += ",";
  features += feature;
}

FailureOr<llvm::SmallVector<char, 0>>
translateTritonToHsaco(ModuleOp module, const TritonToHsacoOptions &options) {
  initializeLLVMTargets();

  // Note: Translation interfaces must be registered before running the pass
  // pipeline. They are registered in:
  // 1. registerTritonToHsacoTranslation() for standalone translation use
  // 2. InitRocMLIRDialects.h for rocmlir-driver and other tools

  // Translate MLIR to LLVM IR (llvm.to_module in compiler.py)
  llvm::LLVMContext llvmContext;
  llvmContext.setDiagnosticHandler(std::make_unique<SuppressWarningHandler>());
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate module to LLVM IR\n";
    return failure();
  }

  StringRef arch = options.arch;
  std::string features = options.features;
  bool enableAsan = (StringRef(options.features).contains("+xnack"));

  // Upstream compiler.py disable_real_true16_feature() passes `-real-true16`
  // for every gfx11 (RDNA3 / RDNA3.5) kernel, unconditionally, in both
  // make_amdgcn and make_hsaco. It is no longer gated on fp8 operand usage.
  bool disableTrue16 = arch.starts_with("gfx11");
  bool enableExpertScheduling =
      isExpertSchedulingEnabled(arch, options.useExpertScheduling);

  auto triple = llvm::Triple(options.triple);
  // Set target triple and data layout (attach_target_triple in compiler.py)
  llvmModule->setTargetTriple(triple);

  // attach_datalayout in compiler.py
  auto tm = createTargetMachine(*llvmModule, triple, arch, features,
                                options.enableFpFusion);
  if (!tm) {
    return failure();
  }
  llvmModule->setDataLayout(tm->createDataLayout());

  // First screen, on the unoptimized IR. It exists because reaching the
  // optimized module is itself expensive on exactly the configs worth
  // rejecting: some cases spend 33s in optimizeModule() alone, so waiting
  // until after it throws away most of what the check is meant to save.
  // Only peak width is screened here. Carried width is not usable this early:
  // how much of it optimization removes depends on the kernel, from about half
  // for a convolution to all but a tenth for an attention, so there is no
  // bound on the unoptimized count that means the same thing for both.
  constexpr uint64_t kPeakPercent = 700;
  if (failed(checkRegisterPressure(module, *llvmModule, tm.get(), arch,
                                   kPeakPercent, /*carriedPercent=*/0,
                                   "unoptimized")))
    return failure();

  // Set AMD-specific control constants
  setISAVersion(*llvmModule, arch);
  setABIVersion(*llvmModule, 500);

  int waveSize = rock::getWaveSize(arch);
  addControlConstant(*llvmModule, "__oclc_finite_only_opt", 8, 0);
  addControlConstant(*llvmModule, "__oclc_correctly_rounded_sqrt32", 8, 1);
  addControlConstant(*llvmModule, "__oclc_unsafe_math_opt", 8, 0);
  addControlConstant(*llvmModule, "__oclc_wavefrontsize64", 8, waveSize == 64);

  int numWarps = options.numWarps;
  if (auto totalNumWarps =
          module->getAttrOfType<IntegerAttr>("ttg.total-num-warps")) {
    if (numWarps != totalNumWarps.getInt()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "ttg.total-num-warps != rock.num_waves ("
                 << totalNumWarps.getInt() << " != " << numWarps << ")\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "This can happen due to warp-specialization\n");
    }
    numWarps = totalNumWarps.getInt();
  }

  int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(module);
  if (numCTAs != options.numCTAs) {
    LLVM_DEBUG(llvm::dbgs()
               << "numCTAs mismatch: TritonGPUDialect::getNumCTAs=" << numCTAs
               << " vs options.numCTAs=" << options.numCTAs << "\n");
  }

  // Set kernel attributes
  setKernelAttributes(*llvmModule, arch, features, numWarps, options.wavesPerEU,
                      numCTAs, options.allowFlushDenorm, enableAsan,
                      enableExpertScheduling, options.llvmFnAttrs);

  // Link external device libraries (ocml.bc, ockl.bc, asanrtl.bc, etc.)
  // compiler.py lines 412-423
  // Auto-detect needed libraries by scanning for unresolved __ocml_/__ockl_
  // references (same logic as need_extern_lib in triton_amd.cc).
  std::vector<std::string> libPaths = options.externLibPaths;
  {
    auto needsLib = [&](StringRef libName) -> bool {
      for (llvm::Function &f : *llvmModule) {
        if (f.hasExternalLinkage() && f.hasName() && !f.hasExactDefinition()) {
          if (f.getName().contains(libName))
            return true;
        }
      }
      return false;
    };
    // Triton bundles device libraries alongside its backend Python code.
    // Use that path first, fall back to the ROCm system path.
    std::array<std::string, 2> searchDirs = {
        TRITON_AMD_BACKEND_LIB_DIR, // from CMake:
                                    // triton/third_party/amd/backend/lib
        "/opt/rocm/amdgcn/bitcode"  // system fallback
    };
    for (const char *lib : {"ocml", "ockl"}) {
      if (!needsLib(lib))
        continue;
      std::string filename = std::string(lib) + ".bc";
      for (const std::string &dir : searchDirs) {
        std::string path = dir + "/" + filename;
        if (llvm::sys::fs::exists(path)) {
          libPaths.push_back(path);
          break;
        }
      }
    }
  }
  if (!libPaths.empty()) {
    if (!linkExternLibs(*llvmModule, libPaths)) {
      llvm::errs() << "Failed to link external libraries\n";
      return failure();
    }
  }

  std::optional<llvm::OptimizationLevel> optLevel =
      mapToLevel(options.optLevel);
  if (!optLevel.has_value()) {
    llvm::errs() << "Invalid optimization level: " << options.optLevel << "\n";
    return failure();
  }

  // optimize_module in llvm.cc
  optimizeModule(*llvmModule, tm.get(), arch, optLevel.value(), enableAsan);

  // Handle architected SGPRs (compiler.py lines 427-434)
  if (hasArchitectedSGPRs(triple, arch)) {
    for (llvm::Function &fn : *llvmModule) {
      if (!fn.isDeclaration() && fn.hasExternalLinkage()) {
        fn.removeFnAttr("amdgpu-no-workgroup-id-x");
        fn.removeFnAttr("amdgpu-no-workgroup-id-y");
        fn.removeFnAttr("amdgpu-no-workgroup-id-z");
        break;
      }
    }
  }

  // scalarize_packed_fops (compiler.py line 436-437)
  if (options.scalarizePackedFops) {
    for (llvm::Function &fn : *llvmModule) {
      if (!fn.isDeclaration() && fn.hasExternalLinkage()) {
        mlir::triton::AMD::runScalarizePackedFOpsPass(fn);
        break;
      }
    }
  }

  // cleanup_bitcode_metadata in compiler.py
  cleanupBitcodeMetadata(*llvmModule);

  // disable_print_inline in compiler.py
  disablePrintInline(*llvmModule);

  // make_amdgcn (compiler.py)
  std::string asmFeatures;
  if (disableTrue16)
    asmFeatures = "-real-true16";
  auto tmAsm = createTargetMachine(*llvmModule, triple, arch, asmFeatures,
                                   options.enableFpFusion);
  if (!tmAsm) {
    return failure();
  }

  // Dump LLVM IR if LLVM_IR_ENABLE_DUMP is set (matches upstream Triton's
  // env var name; see external/triton/include/triton/Tools/Sys/GetEnv.h).
  if (const char *dumpEnv = std::getenv("LLVM_IR_ENABLE_DUMP")) {
    std::string envVal(dumpEnv);
    if (envVal == "1") {
      llvm::errs() << "// -----// LLVM IR Dump //----- //\n";
      llvmModule->print(llvm::errs(), nullptr);
      llvm::errs() << "\n";
    }
  }

  // Second screen, on the optimized IR, which is the faithful measure: it is
  // what the register allocator will actually be handed, and the only point
  // where carried width can be trusted.
  constexpr uint64_t kCarriedPercent = 150;
  if (failed(checkRegisterPressure(module, *llvmModule, tmAsm.get(), arch,
                                   kPeakPercent, kCarriedPercent, "optimized")))
    return failure();

  std::string amdgcnAsm = makeAMDGCN(*llvmModule, tmAsm.get());
  if (amdgcnAsm.empty()) {
    llvm::errs() << "Failed to generate AMDGCN assembly\n";
    return failure();
  }

  // LLVMContext::diagnose no longer aborts on a DS_Error diagnostic; it only
  // records DiagnosticHandler::HasErrors and prints the message. Backend
  // errors such as the AMDGPU RegisterAllocator out-of-registers error surface
  // this way during code generation, so propagate them as a failure instead of
  // emitting a binary for a kernel that the backend rejected.
  if (llvmContext.getDiagHandlerPtr()->HasErrors) {
    llvm::errs() << "LLVM backend reported errors during code generation\n";
    return failure();
  }

  // make_amdgcn (compiler.py)
  if (const char *dumpEnv = std::getenv("AMDGCN_ENABLE_DUMP")) {
    std::string envVal(dumpEnv);
    if (envVal == "1") {
      llvm::errs() << "// -----// AMDGCN Dump //----- //\n"
                   << amdgcnAsm << "\n";
    }
  }

  // make_hsaco (compiler.py)
  std::string hsacoFeatures;
  if (enableAsan)
    hsacoFeatures = "+xnack";
  if (disableTrue16)
    appendFeature(hsacoFeatures, "-real-true16");
  auto hsaco = makeHSACO(amdgcnAsm, triple, arch, hsacoFeatures);
  if (!hsaco) {
    return failure();
  }

  return llvm::SmallVector<char, 0>(hsaco->begin(), hsaco->end());
}

void registerTritonToHsacoTranslation() {
  TranslateFromMLIRRegistration registration(
      "triton-to-hsaco", "Translate Triton LLVM IR to HSACO binary",
      [](ModuleOp module, raw_ostream &output) {
        // Default options - in practice these would come from command line
        TritonToHsacoOptions options;
        auto hsacoOrErr = translateTritonToHsaco(module, options);
        if (failed(hsacoOrErr))
          return failure();
        output.write(hsacoOrErr->data(), hsacoOrErr->size());
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect>();
        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerGPUDialectTranslation(registry);
        mlir::registerROCDLDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);
      });
}

} // namespace rock
} // namespace mlir

//===----------------------------------------------------------------------===//
// Pass Wrapper
//===----------------------------------------------------------------------===//

namespace mlir {
namespace rock {

#define GEN_PASS_DEF_TRITONTOHSACOPASS
#include "mlir/Dialect/Rock/Passes.h.inc"

namespace {

/// Pass wrapper that calls the TritonToHsaco translation.
/// This allows the translation to be used in pass pipelines.
class TritonToHsacoPass
    : public impl::TritonToHsacoPassBase<TritonToHsacoPass> {
public:
  using TritonToHsacoPassBase::TritonToHsacoPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Build options from pass parameters
    TritonToHsacoOptions options;
    options.triple = triple.getValue();
    options.arch = arch.getValue();
    options.features = features.getValue();
    options.optLevel = optLevel.getValue();
    options.numWarps = numWarps.getValue();
    options.numCTAs = numCTAs.getValue();
    options.wavesPerEU = wavesPerEU.getValue();
    options.enableFpFusion = enableFpFusion.getValue();
    options.allowFlushDenorm = allowFlushDenorm.getValue();
    options.scalarizePackedFops = scalarizePackedFops.getValue();
    options.llvmFnAttrs = llvmFnAttrs.getValue();
    options.useExpertScheduling = useExpertScheduling.getValue();

    // Call the translation
    auto hsacoOrErr = translateTritonToHsaco(module, options);
    if (failed(hsacoOrErr)) {
      signalPassFailure();
      return;
    }

    // Store the HSACO binary as a module attribute
    llvm::SmallVector<char, 0> &hsaco = *hsacoOrErr;
    auto hsacoAttr = StringAttr::get(module.getContext(),
                                     StringRef(hsaco.data(), hsaco.size()));
    module->setAttr("triton.hsaco", hsacoAttr);
  }
};

} // namespace
} // namespace rock
} // namespace mlir
