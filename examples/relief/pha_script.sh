mpiexec -n 9 python -m mpisppy.generic_cylinders --module-name relief --solver-name xpress_persistent --sep-rho --lagrangian --xhatshuffle --max-iterations 100 --sep-rho-multiplier 0.5 --rel-gap 1e-4 --intra-hub-conv-thresh 1e-7 --integer-relax-then-enforce --integer-relax-then-enforce-ratio=0.9 --reduced-costs-w-init=0.1


