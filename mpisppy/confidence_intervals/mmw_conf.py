# This software is distributed under the 3-clause BSD License.

# To test: (from confidence_intervals directory; assumes the npy file is in the farmer directory) 
# python mmw_conf.py mpisppy/tests/examples/farmer.py --xhatpath ../../examples/farmer/farmer_root_nonants.npy --confidence-level 0.95 --EF-solver-name gurobi --MMW-num-batches 3 --MMW-batch-size 5

import re
import sys
from mpisppy.utils import config
from pyomo.common.fileutils import import_file
from mpisppy.utils.sputils import option_string_to_dict
from mpisppy.confidence_intervals import ciutils
from mpisppy.confidence_intervals import mmw_ci
from mpisppy import global_toc


if __name__ == "__main__":
    global_toc("Start mmw_conf")
    cfg = config.Config()
    # args for mmw part of things
    cfg.add_to_config('xhatpath',
                         domain=str,
                         description="path to .npy file with feasible nonant solution xhat",
                         default=".")
    cfg.add_to_config('EF_solver_name',
                         description="name of solver to be used",
                         domain=str,
                         default='')
    cfg.add_to_config('confidence_level', 
                         description="defines confidence interval size (default 0.95)", 
                         domain=float,
                         default=None) #None will set alpha = 0.95 
    cfg.add_to_config('start_scen',
                         description="starting scenario number (perhpas to avoid scenarios used to get solve xhat) default 0",
                         domain=int,
                         default=0)
    cfg.add_to_config("MMW_num_batches",
                         description="number of batches used for MMW confidence interval (default 2)",
                         domain=int,
                         default=2)
    cfg.add_to_config("MMW_batch_size",
                         description="batch size used for MMW confidence interval, if None then batch_size = num_scens (default to None)",
                         domain=int,
                         default=None) #None means take batch_size=num_scens
    cfg.add_to_config("solver_options",
                         description="space separated string of solver options, e.g. 'option1=value1 option2 = value2'",
                         domain=str,
                         default='')

    # now get the extra args from the module
    mname = sys.argv[1]  # will be assigned to the model_module_name config arg
    try:
        m = import_file(mname)
    except:
        try:
            m = import_file(f"{mname}.py")
        except:
            raise RuntimeError(f"Could not import module: {mname}")


    m.inparser_adder(cfg)
    # the inprser_adder might want num_scens, but mmw contols the number of scenarios
    try:
        del cfg["num_scens"] 
    except:
        pass

    parser = cfg.create_parser("mmw_conf")
    # the module name is very special because it has to be plucked from argv

    parser.add_argument(
            "model_module_name", help="amalgamator compatible module (often read from argv)", type=str,
        )
    cfg.add_to_config("model_module_name",
                         description="amalgamator compatible module",
                         domain=str,
                         default='',
                         argparse=False)

    cfg.model_module_name = mname
    
    args = parser.parse_args()  # from the command line
    args = cfg.import_argparse(args)
    
    #parses solver options string
    solver_options = option_string_to_dict(cfg.solver_options)

    # convert instance path to module name:
    modelpath = re.sub('/','.', cfg.model_module_name)
    modelpath = re.sub(r'\.py','', modelpath)

    # Read xhats from xhatpath
    xhat = ciutils.read_xhat(cfg.xhatpath)

    if cfg.MMW_batch_size == None:
        raise RuntimeError("mmw_conf requires MMW_batch_size")

    refmodel = modelpath #Change this path to use a different model

    cfg.quick_assign("EF_2stage", bool, True)  # all this script supports for now

    num_batches = cfg.MMW_num_batches
    batch_size = cfg.MMW_batch_size

    mmw = mmw_ci.MMWConfidenceIntervals(refmodel, cfg, xhat, num_batches, batch_size=batch_size, start = cfg.start_scen,
                       verbose=True)

    cl = float(cfg.confidence_level)

    #objective_gap removed sept29th 2022
    r = mmw.run(confidence_level=cl)

    global_toc(r)
    
