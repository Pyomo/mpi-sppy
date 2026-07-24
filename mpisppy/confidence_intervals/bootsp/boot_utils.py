###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Utilities shared by the bootstrap confidence-interval code.

import json
import enum
import inspect
import importlib
import mpisppy.utils.config as config

# The mpi-sppy MPI shim lets Windows users run this code without MPI.
# These communicators are defined once here and imported by the estimator
# and driver modules so there is a single source of truth.
import mpisppy.MPI as MPI

comm = MPI.COMM_WORLD
n_proc = comm.Get_size()
my_rank = comm.Get_rank()
rankcomm = comm.Split(key=my_rank, color=my_rank)  # a private single-rank comm


class BootMethods(enum.Enum):
    Classical_gaussian = "Classical_gaussian"
    Classical_quantile = "Classical_quantile"
    Extended = "Extended"
    Subsampling = "Subsampling"
    Bagging_with_replacement = "Bagging_with_replacement"
    Bagging_without_replacement = "Bagging_without_replacement"
    Smoothed_boot_epi = "Smoothed_boot_epi"
    Smoothed_boot_kernel = "Smoothed_boot_kernel"
    Smoothed_boot_epi_quantile = "Smoothed_boot_epi_quantile"
    Smoothed_boot_kernel_quantile = "Smoothed_boot_kernel_quantile"
    Smoothed_bagging = "Smoothed_bagging"

    @classmethod
    def has_member_key(cls, key):
        return key in cls.__members__

    @classmethod
    def list_of_members(cls):
        return list(cls.__members__.keys())

    @classmethod
    def check_for_it(cls, key):
        if not cls.has_member_key(key):
            raise ValueError(f"Token={key} was not found in list={cls.list_of_members()}")


def is_smoothed(boot_method):
    """ True if boot_method names one of the smoothed methods. """
    return "Smoothed" in boot_method


def empirical_members():
    """ The BootMethods tokens that are available now (the empirical ones). """
    return [m for m in BootMethods.list_of_members() if not is_smoothed(m)]


def smoothed_not_yet_merged(boot_method):
    """ Raise a friendly error for a smoothed method that is not merged yet.

    The smoothed bootstrap methods depend on the statdist distribution
    library and are being merged separately. Until they land, the empirical
    methods are available here and the full set lives in the boot-sp package.
    """
    raise RuntimeError(
        f"boot_method={boot_method} is a smoothed method, which is not yet "
        "available in mpi-sppy (it arrives in a follow-on merge along with "
        "the statdist distribution library). Use one of the empirical "
        f"methods {empirical_members()} here, or the smoothed methods in the "
        "separate boot-sp package (https://github.com/boot-sp/boot-sp)."
    )


def module_name_to_module(module_name):
    if inspect.ismodule(module_name):
        module = module_name
    else:
        module = importlib.import_module(module_name)
    return module


def cfg_for_boot():
    """ Create and return a Config object for the bootstrap code.

    Returns:
        cfg (Config): the config object with the bootstrap options added
    """
    cfg = config.Config()
    # module name gets special parsing
    cfg.add_to_config(name="module_name",
                      description="file name that had scenario creator, etc.",
                      domain=str,
                      default=None,
                      argparse=False)
    cfg.add_to_config(name="max_count",
                      description="The total sample size=M+N",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="candidate_sample_size",
                      description="Sample size to for xhat=M",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="sample_size",
                      description="Sample size for bootstrap=N",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="subsample_size",
                      description="Bagging subsample_size",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="nB",
                      description="number of boot/bag samples",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="alpha",
                      description="significance level two-tailed (e.g. 0.05)",
                      domain=float,
                      default=None)
    cfg.add_to_config(name="seed_offset",
                      description="For some instances this enables replication.",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="optimal_fname",
                      description="(optional for simulations) the name of a npy file with pre-stored optimal; use 'None' when not present",
                      domain=str,
                      default=None)
    cfg.add_to_config(name="xhat_fname",
                      description="(optional) the name of an npy file with a pre-stored xhat; use 'None' when not present",
                      domain=str,
                      default=None)
    cfg.add_to_config(name="solver_name",
                      description="name of solver (e.g. gurobi_direct)",
                      domain=str,
                      default=None)
    cfg.add_to_config(name="boot_method",
                      description="A token naming the boot method (e.g Bagging_with_replacement)",
                      domain=str,
                      default=None)
    cfg.add_to_config(name="trace_fname",
                      description="(optional) the name of an output file that will be appended to by the simulation code; use 'None' when not present",
                      domain=str,
                      default=None,
                      argparse=False)
    cfg.add_to_config(name="coverage_replications",
                      description="number of replications for simulating to get coverage rate.",
                      domain=int,
                      default=None,
                      argparse=False)
    cfg.add_to_config(name="smoothed_B_I",
                      description="number of initial fixed points to use in smoothed bagging.",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="smoothed_center_sample_size",
                      description="number of points to sample from the fitted distribution for the gap center.",
                      domain=int,
                      default=None)
    return cfg


def _process_module(mname):
    # factored code
    module = module_name_to_module(mname)
    cfg = cfg_for_boot()
    assert hasattr(module, "inparser_adder"), \
        f"The module {mname} must have the inparser_adder function"
    module.inparser_adder(cfg)
    assert len(cfg) > 0, f"cfg is empty after inparser_adder in {mname}"
    return cfg


def cfg_from_json(json_fname):
    """ create a config object for the bootstrap code from a json file
    Args:
        json_fname (str): json file name, perhaps with path
    Returns:
        cfg (Config object): populated Config object
    Note:
        Used by the simulation code
    """
    try:
        with open(json_fname, "r") as read_file:
            options = json.load(read_file)
    except Exception:
        print(f"Could not read the json file: {json_fname}")
        raise
    assert "module_name" in options, "The json file must include module_name"
    cfg = _process_module(options["module_name"])

    badtrip = False

    def _dobool(idx):
        nonlocal badtrip
        if idx not in options:
            badtrip = True
            # such an index will raise two complaints...
            print(f"ERROR: {idx} must be in json {json_fname}")
            return
        if options[idx].lower().capitalize() == "True":
            options[idx] = True
        elif options[idx].lower().capitalize() == "False":
            options[idx] = False
        else:
            badtrip = True
            print(f"ERROR: Needed 'True' or 'False', got {options[idx]} for {idx}")

    # get every cfg index from the json
    for idx in cfg:
        if idx not in options:
            if "smoothed" in idx and "Smoothed" not in cfg.boot_method:
                continue
            badtrip = True
            print(f"ERROR: {idx} not in the options read from {json_fname}")
            continue
        if options[idx] != "None":
            # TBD: query the cfg to see if it is bool
            if str(options[idx]).lower().capitalize() == "True" or str(options[idx]).lower().capitalize() == "False":
                _dobool(idx)  # do not return options, just modify cfg
            cfg[idx] = options[idx]
        else:
            cfg[idx] = None

    BootMethods.check_for_it(options["boot_method"])
    if badtrip:
        raise RuntimeError(f"There were missing options in the json file: {json_fname}")
    else:
        return cfg


def cfg_from_parse(module_name, name=None):
    """ create a config object for the bootstrap code by parsing the command line
    Args:
        module_name (str): name of module with scenario creator and helpers
        name (str): name for parser on the command line (e.g. user_boot)
    Returns:
        cfg (Config object): Config object populated by parsing the command line
    """

    cfg = _process_module(module_name)

    parser = cfg.create_parser(name)
    # the module name is very special because it has to be plucked from argv
    parser.add_argument(
            "module_name", help="amalgamator compatible module (often read from argv)", type=str,
        )
    cfg.module_name = module_name

    args = parser.parse_args()  # from the command line
    args = cfg.import_argparse(args)

    return cfg


def compute_xhat(cfg, module):
    """ Use the module's xhat generator to find an xhat (used by the drivers).
    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
    Returns:
        xhat (dict): the optimal nonants in the format specified by mpi-sppy
    Note:
        The code to solve for xhat must be provided by the module. mpi-sppy's
        confidence-interval code uses the fixed name ``xhat_generator``; the
        boot-sp convention was the module-specific ``xhat_generator_<module>``.
        We look for the fixed name first and fall back to the legacy name.

        Only rank 0 calls the generator; the result is broadcast so that
        every rank works with the same xhat even when the problem has
        alternative optima or the solver is nondeterministic.
    """
    fixed_name = "xhat_generator"
    legacy_name = f"xhat_generator_{cfg.module_name}"
    xhat_fct = getattr(module, fixed_name, None)
    if xhat_fct is None:
        xhat_fct = getattr(module, legacy_name, None)
    if xhat_fct is None:
        raise RuntimeError(
            f"\nModule {cfg.module_name} must contain a function "
            f"'{fixed_name}' (or the legacy '{legacy_name}') "
            "when xhat_fname is not given")
    if not hasattr(module, "kw_creator"):
        raise RuntimeError(f"\nModule {cfg.module_name} must contain a function "
                           "kw_creator when xhat_fname is not given")
    if not hasattr(module, "scenario_names_creator"):
        raise RuntimeError(f"\nModule {cfg.module_name} must contain a function "
                           "scenario_names_creator when xhat_fname is not given")
    # Computing xhat_k
    xhat_scenario_names = module.scenario_names_creator(cfg.candidate_sample_size,
                                                        start=cfg.sample_size)

    xgo = module.kw_creator(cfg)
    xgo.pop("solver_name", None)  # it will be given explicitly
    xgo.pop("num_scens", None)
    xgo.pop("scenario_names", None)  # given explicitly
    if my_rank == 0:
        xhat_k = xhat_fct(xhat_scenario_names, solver_name=cfg.solver_name, **xgo)
    else:
        xhat_k = None
    if n_proc > 1:
        xhat_k = comm.bcast(xhat_k, root=0)
    return xhat_k


def check_BFs(cfg):
    BFs = cfg.get("branching_factors") or [0]
    if len(BFs) > 1:
        raise ValueError("Only two-stage problems are presently supported.\n"
                         f"branching_factors was {BFs}")


if __name__ == "__main__":
    print("boot_utils does not have a main program.")
