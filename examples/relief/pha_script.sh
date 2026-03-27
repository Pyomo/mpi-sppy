mpiexec -n 9 python -m mpisppy.generic_cylinders --module-name relief --solver-name gurobi --sep-rho --lagrangian --xhatshuffle --max-iterations 800 --sep-rho-multiplier 0.07 --rel-gap 1e-4 --intra-hub-conv-thresh 1e-7

