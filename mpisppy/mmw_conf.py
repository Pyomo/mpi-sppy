# This software is distributed under the 3-clause BSD License.
# write stand alone MMW program mmw_conf.py 
# (look at bottom half of mmw_ci)

# Look at Main of MMW
# (what is an .npy file?)
# python mmw_conf.py instance.py xhat.npy solver

# python mmw_conf.py aircond_submodels.py ac.npy gurobids
# python mmw_conf.py mpisppy/tests/examples/farmer.py ../examples/farmer/farmer_root_nonants.npy gurobi

# Need to clean this up, as Dr. Woodruff for help here

import sys
import argparse
import mpisppy.utils.amalgomator as ama
import mpisppy.utils.xhat_eval as xhat_eval
import mpisppy.utils.sputils as sputils
import re
from mpisppy.confidence_intervals import mmw_ci

mmw_conf, instance, xhatpath, solver_name = sys.argv

print('instance is: ', instance)
print('xhat is: ', xhatpath)
print('solver is: ', solver_name)


if __name__ == "__main__":

    # parse instance path:
    modelpath = re.sub('/','.', instance)
    modelpath = modelpath[:-3]

    refmodel = modelpath #Change this path to use a different model
    #Compute the nonant xhat (the one used in the left term of MMW (9) ) using
    #                                                        the first scenarios
    print("reference model is: ", refmodel)
    
    ama_options = {"EF-2stage": True,# 2stage vs. mstage
               "start": False,
               "EF_solver_name": solver_name,
               "num_scens": 3}   #Are the scenario shifted by a start arg ?
    
    #add information about batches
    # ama_extraargs = argparse.ArgumentParser(add_help=False)
    # ama_extraargs.add_argument("--MMW-num-batches",
    #                         help="number of batches used for MMW confidence interval (default 1)",
    #                         dest="num_batches",
    #                         type=int,
    #                         default=1)
    
    # ama_extraargs.add_argument("--MMW-batch-size",
    #                         help="batch size used for MMW confidence interval (default None)",
    #                         dest="batch_size",
    #                         type=int,
    #                         default=None) #None means take batch_size=num_scens
    
    ama_object = ama.from_module(refmodel, ama_options, use_command_line=False)

    options = ama_object.options
    options['solver_options'] = options['EF_solver_options']
    xhat = mmw_ci.read_xhat(xhatpath)
   
    #should we except these at arguments?
    num_batches = 3#ama_object.options['num_batches']
    batch_size = 3 #ama_object.options['batch_size']
    
    mmw = mmw_ci.MMWConfidenceIntervals(refmodel, options, xhat, num_batches,batch_size=batch_size,
                       verbose=False)
    r=mmw.run()

    print(mmw.result)