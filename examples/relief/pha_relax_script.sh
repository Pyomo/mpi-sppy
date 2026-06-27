mpiexec -n 8 python -m mpisppy.generic_cylinders --module-name relief --solver-name gurobi --lagrangian --xhatshuffle --max-iterations 800 --rel-gap 1e-5 --intra-hub-conv-thresh 1e-7 --ph-primal --relaxed-ph --relaxed-ph-rescale-rho-factor 0.8 --default-rho 2.0
# --default-rho 2.0
# --sep-rho
