#!/usr/bin/env perl
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
use 5.26.0;
use strict;
use warnings;

while (<>) {
    s/([vsa])\d+/$1?/g;
    s/([vsa])\[\d+:\d+\]/$1?/g;
    print;
}
