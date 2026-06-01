mpiexec -n 6 python -m mpisppy.generic_cylinders --module-name relief --solver-name highs --sep-rho --lagrangian --xhatshuffle --max-iterations 800 --sep-rho-multiplier 1.0 --rel-gap 1e-4 --intra-hub-conv-thresh 1e-7 --max-stalled-iters 1500 --linearize-proximal-terms

