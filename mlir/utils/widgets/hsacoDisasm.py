#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# Portions derived from LLVM's StringExtras.
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Disassemble the `triton.hsaco` kernel binary embedded in a rocmlirTriton
# module. This is the reverse of hsacoToStringAttr.py: that one escapes an hsaco
# ELF into rocmlirTriton serialized text, this one recovers the ELF from that
# text and hands it to llvm-objdump.
#
# Only the instruction text is printed, so that two binaries can be compared
# without tripping over differences that do not affect the kernel. Addresses,
# instruction encodings and symbol names are dropped.
import argparse
import shutil
import re
import subprocess
import sys
import tempfile

ATTR_START = re.compile(r'triton\.hsaco = "')


# The inverse of print_escaped_string in hsacoToStringAttr.py, which follows
# https://github.com/llvm/llvm-project/blob/dc37dc824aabbbe3d029519f43f0b348dcad7027/llvm/lib/Support/StringExtras.cpp#L62-L71
# A backslash is doubled, printable characters other than `"` are literal, and
# everything else is a `\XX` hex pair. Note that this format has no `\n` or `\t`
# form; those bytes come through as hex like any other non-printable.
def parse_escaped_string(text, pos):
    out = bytearray()
    while pos < len(text):
        char = text[pos]
        if char == '"':  # an unescaped quote ends the attribute
            return bytes(out)
        if char != '\\':
            out.extend(char.encode('utf-8', errors='surrogateescape'))
            pos += 1
        elif text[pos + 1] == '\\':
            out.append(0x5C)
            pos += 2
        else:
            out.append(int(text[pos + 1:pos + 3], 16))
            pos += 3
    sys.exit("unterminated triton.hsaco attribute")


def extract_hsaco(path):
    with open(path, encoding='utf-8', errors='surrogateescape') as f:
        text = f.read()

    start = ATTR_START.search(text)
    if start is None:
        sys.exit(f"{path}: no triton.hsaco attribute found")

    elf = parse_escaped_string(text, start.end())
    if not elf.startswith(b"\x7fELF"):
        sys.exit(f"{path}: recovered blob is not an ELF")
    return elf


def disassemble(elf):
    objdump = shutil.which("llvm-objdump")
    if objdump is None:
        sys.exit("llvm-objdump not found on PATH")

    # llvm-objdump reads the AMDGPU target out of the ELF, so no -mcpu here.
    with tempfile.NamedTemporaryFile(suffix=".hsaco") as tmp:
        tmp.write(elf)
        tmp.flush()
        return subprocess.run([objdump, "-d", tmp.name], check=True, capture_output=True,
                              text=True).stdout


def instruction_text(disasm):
    lines = []
    for line in disasm.splitlines():
        # Instructions are indented; headers and `<symbol>:` labels are not.
        if not line[:1].isspace():
            continue
        text = line.split("//")[0].strip()
        if text:
            lines.append(text)
    return lines


def add_args():
    parser = argparse.ArgumentParser(
        description="Disassemble the hsaco kernel embedded in a rocmlirTriton "
        "module.")

    parser.add_argument("-i",
                        help="Input rocmlirTriton module with a "
                        "triton.hsaco attribute",
                        required=True)
    parser.add_argument("-o", help="Output disassembly file (default stdout)", default=None)

    return parser.parse_args()


def main(args):
    instructions = instruction_text(disassemble(extract_hsaco(args.i)))
    if not instructions:
        sys.exit(f"{args.i}: disassembly is empty")

    out = open(args.o, 'w') if args.o else sys.stdout
    print("\n".join(instructions), file=out)


if __name__ == "__main__":
    main(add_args())
