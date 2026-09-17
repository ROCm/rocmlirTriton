#!/usr/bin/env python3

# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from pathlib import Path
from typing import List


def generate(output_path: Path, library_path: Path, libraries: List[str]) -> None:
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for library in libraries:
            bitcode_path = library_path / f"{library}.bc"
            if not bitcode_path.is_file():
                raise SystemExit(f"missing device bitcode: {bitcode_path}")
            bitcode = bitcode_path.read_bytes()
            print(
                f"static constexpr unsigned char {library}_bytes[] = {{",
                file=output,
            )
            for index, byte in enumerate(bitcode):
                line_end = "\n" if index % 16 == 15 else " "
                print(f"{byte},", file=output, end=line_end)
            if len(bitcode) % 16 != 0:
                print(file=output)
            print("};", file=output)

        print(
            """static const llvm::StringMap<llvm::StringRef> &getDeviceLibraries() {
  static const llvm::StringMap<llvm::StringRef> allLibs = {""",
            file=output,
        )
        for library in libraries:
            print(
                f'    {{"{library}.bc", llvm::StringRef('
                f"reinterpret_cast<const char *>({library}_bytes), "
                f"sizeof({library}_bytes))}},",
                file=output,
            )
        print(
            """  };
  return allLibs;
}""",
            file=output,
        )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit("usage: TritonAmdDeviceLibsIncGen.py <output> <library-dir> <library>...")
    generate(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])
