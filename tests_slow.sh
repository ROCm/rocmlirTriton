#!/bin/bash -uvx

cd build && ninja check-rocmlir-build-only ci-performance-scripts && cd ..

cd build && \
 LIT_OPTS="-j32" ninja check-rocmlir -j 32
