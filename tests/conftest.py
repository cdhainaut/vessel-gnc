"""Test-session environment: single-threaded BLAS for deterministic solves.

Must run before any NumPy/CasADi import in the test modules. IPOPT
factorizations (MUMPS) and NumPy reductions are deterministic only when the
linear-algebra backends do not schedule work across threads, and on the small
NMPC/MPCC matrices thread spin-up costs several times the solve itself (the
canonical generation tool documents the same contract, see
``docs/validation.md``).
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
