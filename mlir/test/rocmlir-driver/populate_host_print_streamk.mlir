// Host-code + print generation with a stream-K perf_config (v5, streamKMultiple
// = 1; splitKFactor stays 1). Like split-k, the stream-K output is a
// zero-prefilled accumulator, so it is zero-initialized rather than populated as
// a read operand. This checks a stream-K perf_config flows through
// rocmlir-gen -ph -pr for every element type.

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -perf_config="gemm:v5:64,64,64,1,1,4,16,1,2,0,0,1,-1,-1,-1,-1,-1,-1" -ph -pr | FileCheck %s
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -perf_config="gemm:v5:64,64,64,1,1,4,16,1,2,0,0,1,-1,-1,-1,-1,-1,-1" -ph -pr -t f16 | FileCheck %s --check-prefix=F16
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -perf_config="gemm:v5:64,64,64,1,1,4,16,1,2,0,0,1,-1,-1,-1,-1,-1,-1" -ph -pr -t bf16 | FileCheck %s --check-prefix=BF16
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -perf_config="gemm:v5:64,64,64,1,1,4,16,1,2,0,0,1,-1,-1,-1,-1,-1,-1" -ph -pr -t i8 | FileCheck %s --check-prefix=I8

// CHECK-LABEL: func.func @main()
// CHECK-NEXT: %[[filter:.*]] = memref.alloc() : memref<[[GKCYX:[0-9]+]]x[[TYPE:[a-zA-Z0-9]+]]>
// CHECK-NEXT: arith.constant dense{{.*}} : vector<3x{{.*}}>
// CHECK-NEXT: arith.constant {{.*}} : f32
// CHECK-NEXT: vector.insert {{.*}} : {{.*}} into vector<3x{{.*}}>
// CHECK-NEXT: arith.constant {{.*}} : f32
// CHECK-NEXT: vector.insert {{.*}} : {{.*}} into vector<3x{{.*}}>
// CHECK-NEXT: arith.constant {{.*}} : f32
// CHECK-NEXT: vector.insert {{.*}} : {{.*}} into vector<3x{{.*}}>
// CHECK-NEXT: affine.for %[[if:.*]] = 0 to [[GKCYX]]
// CHECK-NEXT: affine.apply {{.*}}(%[[if]])
// CHECK-NEXT: vector.extract
// CHECK-NEXT: memref.store %{{.*}}, %[[filter]][%[[if]]] : memref<[[GKCYX]]x[[TYPE]]>

// CHECK: %[[input:.*]] = memref.alloc() : memref<[[NGCHIWI:[0-9]+]]x[[TYPE]]>
// CHECK-NEXT: arith.constant dense{{.*}} : vector<3x{{.*}}>
// CHECK-NEXT: arith.constant {{.*}} : f32
// CHECK-NEXT: vector.insert {{.*}} : {{.*}} into vector<3x{{.*}}>
// CHECK-NEXT: arith.constant {{.*}} : f32
// CHECK-NEXT: vector.insert {{.*}} : {{.*}} into vector<3x{{.*}}>
// CHECK-NEXT: arith.constant {{.*}} : f32
// CHECK-NEXT: vector.insert {{.*}} : {{.*}} into vector<3x{{.*}}>
// CHECK-NEXT: affine.for %[[ii:.*]] = 0 to [[NGCHIWI]]
// CHECK-NEXT: affine.apply {{.*}}(%[[ii]])
// CHECK-NEXT: vector.extract
// CHECK-NEXT: memref.store %{{.*}}, %[[input]][%[[ii]]] : memref<[[NGCHIWI]]x[[TYPE]]>

// The stream-K output is a zero-prefilled accumulator, so it is zero-initialized
// (single-element init) exactly like split-k, not populated as a read operand.
// CHECK: %[[output:.*]] = memref.alloc() : memref<[[NGKHOWO:[0-9]+]]x[[TYPE]]>
// CHECK-NEXT: arith.constant dense{{.*}} : vector<1x{{.*}}>
// CHECK-NEXT: arith.constant {{.*}} : f32
// CHECK-NEXT: vector.insert {{.*}} : {{.*}} into vector<1x{{.*}}>
// CHECK-NEXT: affine.for %[[io:.*]] = 0 to [[NGKHOWO]]
// CHECK-NEXT: affine.apply {{.*}}(%[[io]])
// CHECK-NEXT: vector.extract
// CHECK-NEXT: memref.store %{{.*}}, %[[output]][%[[io]]] : memref<[[NGKHOWO]]x[[TYPE]]>

// CHECK: call @{{.*}}({{.*}}, {{.*}}, {{.*}}) : (memref<[[GKCYX]]x[[TYPE]]>, memref<[[NGCHIWI]]x[[TYPE]]>, memref<[[NGKHOWO]]x[[TYPE]]>) -> ()
// CHECK-NEXT: memref.cast %{{.*}} : memref<[[NGKHOWO]]x[[TYPE]]> to memref<*x[[TYPE]]>
// CHECK-NEXT: call @printMemrefF32(%{{.*}}) : (memref<*x[[TYPE]]>) -> ()
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[GKCYX]]x[[TYPE]]>
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[NGCHIWI]]x[[TYPE]]>

// F16: memref.alloc() : memref<{{.*}}>
// F16: call @_memcpy_f16_f32_{{[0-9]+}}(%{{.*}}, %{{.*}})
// BF16: memref.alloc() : memref<{{.*}}>
// BF16: call @_memcpy_bf16_f32_{{[0-9]+}}(%{{.*}}, %{{.*}})
// I8: memref.cast %{{.*}} : memref<{{.*}}> to memref<*xi32>
// I8: call @printMemrefI32(%{{.*}}) : (memref<*xi32>) -> ()
// F16: memref.dealloc {{.*}} : memref<{{.*}}>
// BF16: memref.dealloc {{.*}} : memref<{{.*}}>
// I8: memref.dealloc {{.*}} : memref<{{.*}}>
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[NGKHOWO]]x[[TYPE]]>
// CHECK-NEXT: return
