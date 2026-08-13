# Argument-validation smoke tests for parameterSweeps.py. argparse rejects
# these inputs before the script touches ROCm, so the tests run without a
# GPU and stay snappy. Keeps ``_positive_int`` and the positional choices
# honest after refactors.
#
# RUN: not parameterSweeps.py invalid-kind 2>&1 | FileCheck %s --check-prefix=BAD-KIND
# BAD-KIND: invalid choice: 'invalid-kind'
#
# RUN: not parameterSweeps.py --samples 0 conv 2>&1 | FileCheck %s --check-prefix=NONPOSITIVE-SAMPLES
# NONPOSITIVE-SAMPLES: must be > 0
#
# RUN: not parameterSweeps.py --samples=-3 conv 2>&1 | FileCheck %s --check-prefix=NEGATIVE-SAMPLES
# NEGATIVE-SAMPLES: must be > 0
#
# RUN: not parameterSweeps.py --jobs 0 conv 2>&1 | FileCheck %s --check-prefix=NONPOSITIVE-JOBS
# NONPOSITIVE-JOBS: must be > 0
