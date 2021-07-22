# multistage (4-stage) example using aircond model. Can be any number of stages, does not support unbalanced trees
import pyomo.environ as pyo
import numpy as np
import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgomator as amalgomator
from mpisppy import global_toc

# Use this random stream:
aircondstream = np.random.RandomState()


#============================

if __name__ == "__main__":
    bfs = [4,3,2]
    num_scens = np.prod(bfs) #To check with a full tree
    ama_options = { "EF-mstage": True,
                    "EF_solver_name": "gurobi_direct",
                    "num_scens": num_scens,
                    "_mpisppy_probability": 1/num_scens,
                    "BFs":bfs,
                    "mudev":0,
                    "sigmadev":80
                    }
    refmodel = "mpisppy.tests.examples.aircond_submodels" # WARNING: Change this in SPInstances
    #We use from_module to build easily an Amalgomator object
    ama = amalgomator.from_module(refmodel,
                                  ama_options,use_command_line=False)
    ama.run()
    print(f"inner bound=", ama.best_inner_bound)
    print(f"outer bound=", ama.best_outer_bound)
    
    from mpisppy.confidence_intervals.mmw_ci import MMWConfidenceIntervals
    options = ama.options
    options['solver_options'] = options['EF_solver_options']
    xhat = sputils.nonant_cache_from_ef(ama.ef)['ROOT']
   
    
    num_batches = 10
    batch_size = 100
    
    mmw = MMWConfidenceIntervals(refmodel, options, xhat, num_batches,batch_size=batch_size,
                        verbose=False)
    r=mmw.run(objective_gap=True)
    print(r)
