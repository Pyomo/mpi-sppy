# NOTE: num_scens is restricted by the availability of data directories
python -m mpisppy.confidence_intervals.mmw_conf uc_funcs uc_cyl_nonants.spy.npy cplex --MMW-num-batches 5 --MMW-batch-size 10 --UC-count-for-path 100 --alpha 0.9 --start-scen 10
