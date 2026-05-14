// This is a compile-only test -- it intentionally drops rocm-run -- so
// the LIT success criterion is "rocmlir-driver returned 0 within 5 seconds".
// The timeout is what catches the regression: the bug causes
// rocmlir-driver to hang indefinitely. Five seconds is enough headroom for
// a healthy compile of this kernel.
//
// The arch is hardcoded to gfx950 (not %arch) because the bug only
// reproduces when targeting that architecture. The test is compile-only
// (no kernel execution) so it can run on any host.

// RUN: rocmlir-gen --arch gfx950 --num_chiplets 8 --operation gemm -t f16 -out_datatype f32 -g 1 -m 4096 -k 11008 -n 4096 -transA=False -transB=False --perf_config=gemm:v1:16,16,16,1,1,1,32,1,2,0,0 \
// RUN: | timeout 5 rocmlir-driver -c > /dev/null
