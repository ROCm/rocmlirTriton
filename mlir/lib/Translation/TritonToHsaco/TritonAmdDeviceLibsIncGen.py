#!/usr/bin/env python3

# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from pathlib import Path
from typing import List


def generate(output_path: Path, library_path: Path, libraries: List[str]) -> None:
    with output_path.open("w") as output:
        for library in libraries:
            bitcode = (library_path / f"{library}.bc").read_bytes()
            print(f"static constexpr size_t {library}_size = {len(bitcode)};", file=output)
            print(
                """#if defined __GNUC__
__attribute__((aligned(4096)))
#elif defined _MSC_VER
__declspec(align(4096))
#endif""",
                file=output,
            )
            print(
                f"static constexpr char {library}_bytes[{library}_size + 1] = {{",
                file=output,
            )
            for index, byte in enumerate(bitcode):
                line_end = "\n" if index % 8 == 7 else " "
                print(f"static_cast<char>({byte}),", file=output, end=line_end)
            print("0x00};", file=output)

        print(
            "static constexpr std::initializer_list<"
            "std::pair<llvm::StringRef, llvm::StringRef>> allLibList = {",
            file=output,
        )
        for library in libraries:
            print(
                f'{{"{library}.bc", '
                f"llvm::StringRef({library}_bytes, {library}_size)}},",
                file=output,
            )
        print("};", file=output)
        print(
            """static const llvm::StringMap<llvm::StringRef> &getDeviceLibraries() {
  static const llvm::StringMap<llvm::StringRef> allLibs(allLibList);
  return allLibs;
}""",
            file=output,
        )


if __name__ == "__main__":
    generate(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])
