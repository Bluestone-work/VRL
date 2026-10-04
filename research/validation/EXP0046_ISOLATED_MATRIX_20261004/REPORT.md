# EXP0046 isolated development matrix

This directory contains a diagnostic run only. Seven cells were requested,
five 5-second episodes per cell, with one child process per episode. The
5-second horizon cannot clear the complete task, so Safe Success is zero in
every cell and no method comparison is valid.

The process isolation was necessary: native SIGSEGV/SIGILL occurred in several
child processes even at this short horizon. Completed child rows are retained;
crashed seeds are listed in each `*_summary.json` and excluded from the
aggregate. Crash counts were 2/5, 0/5, and 2/5 for two clusters at
`d_min=1,2,4 mm`, and 1/5, 2/5, and 1/5 for three clusters. The single-cluster
baseline had 0/5 crashes. This is a numerical-integrator gate failure, not
evidence that a spacing value or controller is better.

The next required action is to repair and regression-test the reference
transport/observation crash before increasing the horizon, training a policy,
or opening validation/sealed evaluation.
