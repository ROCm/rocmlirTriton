#!/bin/bash
# Collect calibration data for the register-pressure screen in TritonToHsaco.
#
# For every configuration it records what the screen measured (peak lanes,
# carried lanes, the kernel's accumulator, and each bound), what the screen
# would have decided, and what the configuration actually cost to compile. The
# screen runs in reporting mode (ROCMLIR_REGISTER_PRESSURE_REPORT), so nothing
# is rejected and the compile time is the real one even for a configuration the
# screen would have thrown away -- that is the number that says whether the
# verdict was deserved.
#
# Compile only, no GPU needed. Results are appended, so the script can be
# interrupted and rerun: configurations already recorded are skipped.
#
# usage: collectRegisterPressureData.sh [--build DIR] [--out DIR]
#                                       [--archs LIST] [--ops LIST]
#                                       [--timeout SEC]
set -u

BUILD=$PWD/build
OUT=$PWD/register-pressure-data
ARCHS=gfx1100,gfx950
OPS=gemm,conv,conv-migraphx,attention
# A configuration that needs longer than this is already too slow to be worth
# tuning, so the exact number does not matter and waiting for it costs hours
# across a space this size.
TIMEOUT=30

while [[ $# -gt 0 ]]; do
  case $1 in
    --build)   BUILD=$2; shift 2 ;;
    --out)     OUT=$2; shift 2 ;;
    --archs)   ARCHS=$2; shift 2 ;;
    --ops)     OPS=$2; shift 2 ;;
    --timeout) TIMEOUT=$2; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

GEN=$BUILD/bin/rocmlir-gen
DRIVER=$BUILD/bin/rocmlir-driver
for tool in "$GEN" "$DRIVER"; do
  [[ -x $tool ]] || { echo "not found: $tool (pass --build)" >&2; exit 2; }
done
mkdir -p "$OUT"

# Per-architecture machine facts. rocmlir-gen wants them explicitly, and the
# grid size it picks from them affects nothing this script measures, but an
# accurate value keeps the generated problem realistic.
arch_triple() { case $1 in
  gfx950) echo "gfx950:sramecc+:xnack-" ;;
  *)      echo "$1" ;;
esac; }
arch_num_cu()       { case $1 in gfx950) echo 256 ;; *) echo 48 ;; esac; }
arch_num_chiplets() { case $1 in gfx950) echo 8 ;;   *) echo 1 ;;  esac; }

# The problems. One shape per operation, run on both architectures, so that a
# difference in the data is a difference in the target and not in the problem.
# The convolution is the 1x512x8x8 f16 with a 512x512x3x3 filter that exposed
# the pathological compile times in the first place.
write_problem() { # write_problem <op> <arch> <path>
  local op=$1 arch=$2 path=$3
  local triple; triple=$(arch_triple "$arch")
  local cu; cu=$(arch_num_cu "$arch")
  local chiplets; chiplets=$(arch_num_chiplets "$arch")
  case $op in
    gemm)
      "$GEN" --operation gemm -t f16 --arch "$triple" --num_cu "$cu" \
          --num_chiplets "$chiplets" -g 1 -m 2048 -k 2048 -n 2048 \
          > "$path" 2>/dev/null ;;
    conv)
      "$GEN" --operation conv -t f16 --arch "$triple" --num_cu "$cu" \
          --num_chiplets "$chiplets" --fil_layout gnc01 --in_layout ngc01 \
          --out_layout ngc01 --batchsize 1 --in_channels 512 --in_h 8 --in_w 8 \
          --out_channels 512 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 \
          --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 \
          --groupsize 1 > "$path" 2>/dev/null ;;
    attention)
      "$GEN" -operation attention -t f16 --arch "$triple" --num_cu "$cu" \
          --num_chiplets "$chiplets" -g 4 -seq_len_q 1024 -seq_len_k 1024 \
          -num_heads_q 4 -num_heads_kv 4 -head_dim_qk 64 -head_dim_v 64 \
          > "$path" 2>/dev/null ;;
    conv-migraphx)
      # The same convolution through the migraphx entry point, which is how it
      # reached us and which is the space the worst gfx950 case lives in. The
      # screen runs on the output of the highlevel pipeline, so hand it that.
      local src=$path.migraphx.mlir
      cat > "$src" <<EOF
module {
  func.func @mlir_convolution(%arg0: !migraphx.shaped<1x512x8x8xf16, 32768x64x8x1>, %arg1: !migraphx.shaped<512x512x3x3xf16, 4608x9x3x1>) -> !migraphx.shaped<1x512x8x8xf16, 32768x64x8x1> attributes {rock.arch = "$triple", rock.kernel = "mixr", rock.num_chiplets = $chiplets : i64, rock.num_cu = $cu : i64} {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x512x8x8xf16, 32768x64x8x1>, <512x512x3x3xf16, 4608x9x3x1> -> <1x512x8x8xf16, 32768x64x8x1>
    return %0 : !migraphx.shaped<1x512x8x8xf16, 32768x64x8x1>
  }
}
EOF
      "$DRIVER" -kernel-pipeline=migraphx,highlevel "$src" > "$path" 2>/dev/null ;;
    *) echo "unknown op: $op" >&2; return 1 ;;
  esac
  [[ -s $path ]]
}

# Configurations outside the emitted tuning space. These are hand-written tiles
# on the convolution above, and they are the reason this screen exists: on
# gfx1100 kPerBlock=256 took 60s to compile and kPerBlock=128 took 15s, while
# kPerBlock=144 next door takes 0.9s. A space sweep alone never sees them,
# because the space only ever emits numWaves of 4 or 8 and these use 1.
extra_configs() {
  local base=kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1
  local tail=wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1
  local k
  for k in 64 128 144 192 256; do
    echo "gemm:mPerBlock=16,nPerBlock=64,kPerBlock=$k,$base,numStages=1,$tail"
    echo "gemm:mPerBlock=16,nPerBlock=64,kPerBlock=$k,$base,numStages=2,$tail"
  done
}

# One configuration: compile it with the screen reporting instead of rejecting,
# and emit a row. Fields are name=value so that a column added later does not
# invalidate data already collected.
measure() { # measure <arch> <op> <source> <input> <config> <out.tsv>
  local arch=$1 op=$2 source=$3 input=$4 cfg=$5 out=$6
  local log; log=$(mktemp)
  local start; start=$(date +%s.%N)
  ROCMLIR_REGISTER_PRESSURE_REPORT=1 timeout "$TIMEOUT" \
      "$DRIVER" -c --arch="$(arch_triple "$arch")" --perf-config="$cfg" \
      -o /dev/null "$input" >/dev/null 2>"$log"
  local rc=$?
  local end; end=$(date +%s.%N)
  local secs; secs=$(echo "$end - $start" | bc)

  local status=ok
  case $rc in
    0) ;;
    124) status=too-slow ;;   # over the cap, exact time not interesting
    *)   status=err ;;
  esac

  local row
  row=$(printf 'arch=%s\top=%s\tsource=%s\tsecs=%.2f\tstatus=%s' \
        "$arch" "$op" "$source" "$secs" "$status")
  local stage
  for stage in unoptimized optimized; do
    local line
    line=$(grep -m1 "stage=$stage" "$log")
    if [[ -n $line ]]; then
      # Prefix every measured field with its stage: peak=... becomes
      # unoptimized.peak=...
      local field
      for field in peak peakLimit carried accumulator carriedLimit \
                   carriedFloor verdict; do
        local value
        value=$(sed -n "s/.* $field=\([^ ]*\).*/\1/p" <<< "$line")
        row+=$(printf '\t%s.%s=%s' "$stage" "$field" "${value:-NA}")
      done
    fi
  done
  if [[ $status == err ]]; then
    row+=$(printf '\terror=%s' "$(grep -m1 -o 'error: .*' "$log" | cut -c1-80 | tr '\t' ' ')")
  fi
  printf '%s\tconfig=%s\n' "$row" "$cfg" >> "$out"
  rm -f "$log"
  echo "$status"
}

echo "collecting into $OUT (cap ${TIMEOUT}s per config)"
IFS=, read -ra arch_list <<< "$ARCHS"
IFS=, read -ra op_list <<< "$OPS"

for arch in "${arch_list[@]}"; do
  for op in "${op_list[@]}"; do
    out=$OUT/$arch-$op.tsv
    input=$OUT/$arch-$op.input.mlir
    if ! write_problem "$op" "$arch" "$input"; then
      echo "  $arch/$op: could not build the problem, skipping" >&2
      continue
    fi

    mapfile -t configs < <("$GEN" --emit-tuning-space=exhaustive "$input" 2>/dev/null)
    # The hand-written tiles only mean anything on the convolution they were
    # written for. Keep them in their own list so that each row records which
    # list it came from rather than guessing from the config text.
    extras=()
    [[ $op == conv-migraphx ]] && mapfile -t extras < <(extra_configs)

    touch "$out"
    total=$(( ${#configs[@]} + ${#extras[@]} ))
    done_count=0 slow=0
    echo "  $arch/$op: $total configs (${#extras[@]} outside the space) -> $out"
    for source in space extra; do
      if [[ $source == space ]]; then list=("${configs[@]}")
      else list=("${extras[@]+${extras[@]}}"); fi
      for cfg in "${list[@]+${list[@]}}"; do
        [[ -z $cfg ]] && continue
        # Resume: skip anything already recorded for this arch and op.
        if grep -qF "config=$cfg" "$out" 2>/dev/null; then continue; fi
        status=$(measure "$arch" "$op" "$source" "$input" "$cfg" "$out")
        done_count=$((done_count + 1))
        [[ $status == too-slow ]] && slow=$((slow + 1))
        if (( done_count % 100 == 0 )); then
          echo "    $done_count/$total done, $slow over the cap"
        fi
      done
    done
    rejected=$(grep -c 'verdict=reject' "$out" 2>/dev/null || echo 0)
    echo "    finished: $(($(wc -l < "$out"))) rows, $slow over the cap, $rejected would be rejected"
  done
done

echo
echo "done. Collected:"
wc -l "$OUT"/*.tsv 2>/dev/null
echo
echo "Send back the whole $OUT directory (the .tsv files; the .input.mlir files"
echo "are regenerable and can be left out)."
