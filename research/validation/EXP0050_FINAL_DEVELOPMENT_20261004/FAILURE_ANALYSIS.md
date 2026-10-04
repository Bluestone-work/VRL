# EXP0050 numerical failure record

One of 132 requested main validation attempts failed: learned seed 44 at
65,536 steps, scene 1620000011, process return code -11 (SIGSEGV). The full
matrix has 131 completed attempts and retains this failure in its denominator.
The combined numerical gate is blocked and its inferential comparisons remain
empty. No successful replay replaces the failed row.

The Python-level stack ends inside the geodesic contact calculation at
`environments/mca_physical_env.py:995–996`, called by the physical substep
callback. This identifies a frame at failure, not the underlying native cause.
The simulator uses geodesics for physics; this trace is not evidence that a
controller received routes or true body-edge IDs.

A separate unchanged-checkpoint gdb replay completed normally, with removal
0.010874735972474903 and wall contact 175.74659979533504 cluster-seconds.
Gdb returned 1 because its subsequent register query had no running inferior.
There was no captured native fault in this replay; the root cause remains
intermittent and unresolved. The earlier EXP0048 preflight and EXP0046 fault
also remain recorded. Do not call any Python navigation change a native fix.

Raw stack: `../EXP0050_LEARNED64_VAL_20261004/parts/learned_s44_t65536_n3_scene1620000011.stderr.txt`.
Debug manifest and logs: `../EXP0050_NATIVE_DIAGNOSTIC_20261004/`.

No simulator, dependency, CPU setting or evaluation row was changed to make
this failure disappear. More training is not evidence of numerical repair.
