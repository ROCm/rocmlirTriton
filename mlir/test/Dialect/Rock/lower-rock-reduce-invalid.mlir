// RUN: rocmlir-opt -rock-lower-reduce -verify-diagnostics %s

func.func @test_reduce_max_fp8(
    %input: tensor<2x12x12xf8E4M3FNUZ>,
    %output: tensor<2x12x1xf8E4M3FNUZ>) -> tensor<2x12x1xf8E4M3FNUZ> {
  // expected-error@+2 {{failed to legalize operation 'rock.reduce' that was explicitly marked illegal}}
  // expected-error@+1 {{source element type 'f8E4M3FNUZ' does not support atomic_max}}
  %reduced = rock.reduce max %input {axis = 2 : index} : tensor<2x12x12xf8E4M3FNUZ> -> tensor<2x12x1xf8E4M3FNUZ>
  // expected-error@+1 {{failed to set store method and prefill}}
  %result = rock.store %reduced to %output by set : tensor<2x12x1xf8E4M3FNUZ> -> tensor<2x12x1xf8E4M3FNUZ> to tensor<2x12x1xf8E4M3FNUZ>
  return %result : tensor<2x12x1xf8E4M3FNUZ>
}
