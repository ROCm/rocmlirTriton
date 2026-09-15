// Regression for a rocmlir-driver hang on gfx950: with the perf_config
// below this kernel previously caused rocmlir-driver -c to spin
// indefinitely. Compile-only; cross-compiles to gfx950 so the test runs on
// every CI host. The LIT success criterion is "rocmlir-driver returned 0
// within 5 seconds" -- five seconds is enough headroom for a healthy
// compile of this kernel and short enough to catch the hang.

// RUN: rocmlir-gen --arch gfx950 --num_cu 256 --num_chiplets 8 --operation gemm -t f16 -out_datatype f32 -g 1 -m 4096 -k 11008 -n 4096 -transA=False -transB=False --perf_config=gemm:v1:16,16,16,1,1,1,32,1,2,0,0 \
// RUN: | timeout 5 rocmlir-driver -c > /dev/null
