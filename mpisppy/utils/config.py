###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Replace baseparsers.py and enhance functionality.
# A class drived form pyomo.common.config is defined with
#   supporting member functions.
# NOTE: the xxxx_args() naming convention is used by amalgamator.py

""" Notes
The default for all 'with' options is False and we are dropping the with_
       (and we are dropping the `no` side that was in baseparsers.py)
       (so we are also dropping the use of with_)

Now you assemble the args you want and call the create_parser function,
   which returns an argparse object, E.g.:
parser = cfg.create_parser("myprog")
although most program use
cfg.parse_command_line("program_name")
which create the parser and does the parsing.

If you want to add args, you need to call the add_to_config function

If you want a required arg see num_scens_required() in this file.

If you want a positional arg, you have to DIY:
    parser = cfg.create_parser("tester")
    parser.add_argument(
            "num_scens", help="Number of scenarios", type=int,
        )
    args=parser.parse_args(['3', '--max-iterations', '99', '--solver-name', 'cplex'])
    print(f"{args.num_scens =}")
(Note: you can still attach it to a Config object, but that is also DIY)

    cfg.add_to_config("num_scens",
                         description="Number of Scenarios (required, positional)",
                         domain=int,
                         default=-1,
                         argparse=False)   # special
    # final special treatment of num_scens
    cfg.num_scens = args.num_scens


"""

import argparse
import pyomo.common.config as pyofig

# class to inherit from ConfigDict with a name field
class Config(pyofig.ConfigDict):
    # remember that the parent uses slots

    #===============
    def add_to_config(self, name, description, domain, default,
                      argparse=True,
                      complain=False,
                      argparse_args=None):
        """ Add an arg to the self dict.
        Args:
            name (str): the argument name, underscore seperated
            description (str): free text description
            domain (type): see pyomo config docs
            default (domain): value before argparse
            argparse (bool): if True put on command ine
            complain (bool): if True, output a message for a duplicate
            argparse_args (dict): args to pass to argpars (option; e.g. required, or group)
        """
        if name in self:
            if complain:
                print(f"Duplicate {name} will not be added to self.")
                # raise RuntimeError(f"Trying to add duplicate {name} to self.")
        else:
            c = self.declare(name, pyofig.ConfigValue(
                description = description,
                domain = domain,
                default = default))
            if argparse:
                if argparse_args is not None:
                    c.declare_as_argument(**argparse_args)
                else:
                    c.declare_as_argument()


    #===============
    def add_and_assign(self, name, description, domain, default, value, complain=True):
        """ Add an arg to the self dict and assign it a value
        Args:
             name (str): the argument name, underscore separated
            description (str): free text description
            domain (type): see pyomo config docs
            default (domain): probably unused, but here to avoid cut-and-paste errors
            value (domain): the value to assign
            complain (bool): if True, output a message for a duplicate
        """
        if name in self:
            if complain:
                raise RuntimeError(f"Trying to add duplicate {name=} to cfg {value=}")
        else:
            self.add_to_config(name, description, domain, default, argparse=False)
            self[name] = value


    #===============
    def dict_assign(self, name, description, domain, default, value):
        """ mimic dict assignment
        Args:
            name (str): the argument name, underscore separated
            description (str): free text description
            domain (type): see pyomo config docs
            default (domain): probably unused, but here to avoid cut-and-paste errors
            value (domain): the value to assign
        """
        if name not in self:
            self.add_and_assign(name, description, domain, default, value)
        else:
            self[name] = value


    #===============
    def quick_assign(self, name, domain, value):
        """ mimic dict assignment with fewer args
        Args:
            name (str): the argument name, underscore separated
            domain (type): see pyomo config docs
            value (domain): the value to assign
        """
        self.dict_assign(name, f"field for {name}", domain, None, value)


    #===============
    def get(self, name, ifmissing=None):
        """ replcate the behavior of dict get"""
        if name in self:
            return self[name]
        else:
            return ifmissing

    #===============
    def checker(self):
        """Verify that options *selected* make sense with respect to each other
        """
        def _bad_options(msg):
            raise ValueError("Options do not make sense together:\n"
                             f"{msg}")

        # reduced_costs_rho was removed (deprecated): it was never shown to be
        # effective in practice and it does not support flexible (unequal) rank
        # assignments, so a custom driver could be silently burned. The option
        # is kept so its selection fails loudly here rather than being ignored.
        if self.get("reduced_costs_rho"):
            raise ValueError(
                "reduced_costs_rho was deprecated and removed on 2026-06-14 "
                "(reduced-cost rho was not demonstrated to be effective in "
                "practice and does not support flexible rank assignments; see "
                "https://github.com/Pyomo/mpi-sppy/issues/673). Remove "
                "--reduced-costs-rho; consider --grad-rho instead."
            )

        # remember that True is 1 and False is 0
        if (self.get("APH",0) + self.get("subgradient_hub",0) + self.get("fwph_hub",0) + self.get("ph_primal_hub",0) + self.get("lshaped_hub",0) + self.get("cg_hub",0) + self.get("dualcg_hub",0)) > 1:
            _bad_options("Only one hub can be active.")

        # remember that True is 1 and False is 0
        if (self.get("grad_rho") + self.get("sensi_rho") + self.get("coeff_rho") + self.get("sep_rho")) > 1:

            _bad_options("Only one rho setter can be active.")
        if not (self.get("grad_rho")
                or self.get("sensi_rho")
                or self.get("sep_rho")):
            if self.get("dynamic_rho_primal_crit") or self.get("dynamic_rho_dual_crit"):
                _bad_options("dynamic rho only works with an automated rho setter")

        if self.get("ph_primal_hub")\
           and not (self.get("ph_dual") or self.get("relaxed_ph")):
            _bad_options("--ph-primal-hub is used only when there is a cylinder that provideds Ws "
                         "such as --ph-dual or --relaxed-ph")
        
        if self.get("rc_fixer") and not self.get("reduced_costs"):
            _bad_options("--rc-fixer requires --reduced-costs")

        if self.get("hub_only_solver_logs") and not self.get("solver_log_dir"):
            _bad_options("--hub-only-solver-logs requires --solver-log-dir")

        if self.get("cc_indicator_var", None) is not None and not self.get("EF"):
            # A chance constraint Sum_s p_s z_s >= 1-alpha couples all scenarios
            # and is not separable, so it is supported only for the EF (matching
            # PySP). See doc/designs/chance_constraint_design.md.
            _bad_options("--cc-indicator-var (chance constraint) is currently "
                         "supported only with --EF")

        # Slamming options other than the directives file are meaningless
        # without it; require the file so that a run with no slamming options
        # behaves exactly as it does today (total backward compatibility).
        if self.get("slamming_directives_file") is None and (
                self.get("slam_start_iter") is not None
                or self.get("iters_between_slams") is not None):
            _bad_options("slamming options (--slam-start-iter / "
                         "--iters-between-slams) require "
                         "--slamming-directives-file")

    def add_solver_specs(self, prefix=""):
        sstr = f"{prefix}_solver" if prefix else "solver"
        if prefix:
            prefix += " "
        self.add_to_config(f"{sstr}_name",
                            description= f"{prefix}solver name (default None)",
                            domain = str,
                            default=None)

        self.add_to_config(f"{sstr}_options",
                            description= f"{prefix}solver options; space delimited with = for values (default None)",
                            domain = str,
                            default=None)

        self.add_to_config(f"{sstr}_options_file",
                            description= f"{prefix}path to a JSON solver-options file with sections "
                                         "default/iter0/iterk/starting_at_iter (and a 'spokes' sub-block at the "
                                         "top level, naming each spoke's overrides). CLI flags "
                                         "override file entries at the same predicate.",
                            domain = str,
                            default=None)

    def add_mipgap_specs(self, prefix=""):
        sstr = f"{prefix}_" if prefix else ""
        if prefix:
            prefix += " "
        self.add_to_config(f"{sstr}iter0_mipgap",
                            description=f"{prefix}mip gap option for iteration 0 (default None)",
                            domain=float,
                            default=None)

        self.add_to_config(f"{sstr}iterk_mipgap",
                            description=f"{prefix}mip gap option non-zero iterations (default None)",
                            domain=float,
                            default=None)

    def _common_args(self):
        raise RuntimeError("_common_args is no longer used. See comments at top of config.py")

    def popular_args(self):
        self.add_to_config("max_iterations",
                            description="hub max iterations (default 1)",
                            domain=int,
                            default=1)

        self.add_to_config("time_limit",
                            description="hub time limit in seconds (default None)",
                            domain=int,
                            default=None)

        self.add_to_config("solver_log_dir",
                           description="Put solver logs generated by subproblem solves here (default None). "
                           "WARNING: This will create a file for every single subproblem solve.",
                           domain=str,
                           default=None)

        self.add_to_config("hub_only_solver_logs",
                           description="When set with --solver-log-dir, only the hub writes "
                           "solver logs; spokes do not. Requires --solver-log-dir.",
                           domain=bool,
                           default=False)

        self.add_to_config("xhatter_write_iis",
                           description="When an xhatter (incumbent-finder) rejects a candidate "
                           "because a scenario subproblem is infeasible, write an IIS "
                           "(irreducible infeasible set) for the offending subproblem via "
                           "pyomo.contrib.iis. Useful to diagnose models that should have "
                           "(relatively) complete recourse but don't. Fires AT MOST ONCE per "
                           "cylinder (per MPI rank): the (expensive) IIS computation cannot "
                           "repeat. The facility is chosen by --xhatter-iis-method (default "
                           "'auto'). See doc/src/iis.rst for the full story.",
                           domain=bool,
                           default=False)

        self.add_to_config("xhatter_iis_method",
                           description="Which pyomo.contrib.iis facility --xhatter-write-iis "
                           "uses: 'ilp' writes a .ilp file with a commercial solver "
                           "(cplex/gurobi/xpress); 'explanation' uses "
                           "compute_infeasibility_explanation, which works with any solver; "
                           "'auto' (default) picks 'ilp' for a commercial solver, else "
                           "'explanation'. See doc/src/iis.rst.",
                           domain=str,
                           default="auto")

        self.add_to_config("xhatter_iis_dir",
                           description="Directory for IIS files written by --xhatter-write-iis "
                           "(default: current working directory). File names follow the "
                           "--solver-log-dir convention. See doc/src/iis.rst.",
                           domain=str,
                           default=None)

        self.add_to_config("inspect_buffers_on_shutdown",
                           description="When a spoke detects a shutdown signal, run "
                           "mpisppy.debug_utils.buffer_inspect on the SHUTDOWN receive "
                           "buffer and emit a RuntimeWarning with any findings. "
                           "Off by default.",
                           domain=bool,
                           default=False)

        self.add_to_config("warmstart_subproblems",
                           description="Warmstart subproblems from prior solution.",
                           domain=bool,
                           default=False)

        self.add_to_config("turn_off_names_check",
                           description="Turn off the check that nonant variable names "
                           "match across scenarios (automatically turned off for proper bundles).",
                           domain=bool,
                           default=False)

        self.add_to_config("user_warmstart",
                           description="Will pass the user provided solution as a warmstart for "
                           "each initial subproblem solve.",
                           domain=bool,
                           default=False)

        self.add_solver_specs()

        self.add_to_config("seed",
                            description="Seed for random numbers (default is 1134)",
                            domain=int,
                            default=1134)

        self.add_to_config("default_rho",
                            description="Global rho for PH (default None)",
                            domain=float,
                            default=None)

        self.add_to_config("bundles_per_rank",
                            description="REMOVED: loose bundles no longer supported. Use --scenarios-per-bundle.",
                            domain=int,
                            default=0)

        self.add_to_config('verbose',
                              description="verbose output",
                              domain=bool,
                              default=False)

        self.add_to_config('display_progress',
                              description="display progress at each iteration",
                              domain=bool,
                              default=False)

        self.add_to_config('display_convergence_detail',
                              description="display non-anticipative variable convergence statistics at each iteration",
                              domain=bool,
                              default=False)

        self.add_to_config('display_timing',
                              description="display subproblem solve timing (requires exactly one subproblem per rank)",
                              domain=bool,
                              default=False)

        self.add_to_config("max_solver_threads",
                            description="Limit on threads per solver (default None)",
                            domain=int,
                            default=None)

        self.add_to_config("intra_hub_conv_thresh",
                            description="Within hub convergence threshold (default 1e-10)",
                            domain=float,
                            default=1e-10)

        self.add_to_config("trace_prefix",
                            description="Prefix for bound spoke trace files. If None "
                                 "bound spoke trace files are not written.",
                            domain=str,
                            default=None)

        self.add_to_config("incumbent_on_improvement_filename_prefix",
                            description="If set, incumbent (xhat) bound spokes "
                                 "write the first-stage solution to "
                                 "<prefix>_<NNNN>.csv and <prefix>_<NNNN>.npy "
                                 "each time they find a new best inner bound. "
                                 "<NNNN> is a zero-padded counter starting at "
                                 "0000. Default None disables.",
                            domain=str,
                            default=None)

        self.add_to_config("tee_rank0_solves",
                              description="Some cylinders support tee of rank 0 solves."
                              "(With multiple cylinders this could be confusing.)",
                              domain=bool,
                              default=False)

        self.add_to_config("auxilliary",
                            description="Free text for use by hackers (default '').",
                            domain=str,
                            default='')

        self.add_to_config("presolve",
                           description="Run the distributed presolver. "
                           "Performs distributed feasibility-based bounds tightening and optimization-based bounds tightening.",
                           domain=bool,
                           default=False)

        self.add_to_config("rounding_bias",
                           description="When rounding variables to integers, "
                           "add the given value first. (default = 0.0)",
                           domain=float,
                           default=0.0)

        self.add_to_config("config_file",
                           description="Path to file containing config options",
                           domain=str,
                           default='')

    def presolve_args(self):
        self.add_to_config("obbt",
                           description="Use the optimization-based bounds tightening as part of the presolver",
                           domain=bool,
                           default=False,
                           )
        self.add_to_config("full_obbt",
                           description="Run OBBT on *all* variables, not just the nonanticipative variables",
                           domain=bool,
                           default=False,
                          )
        self.add_to_config("obbt_solver",
                           description="OBBT solver",
                           domain=str,
                           default=None,
                           )
        self.add_to_config("obbt_solver_options",
                           description="OBBT solver options",
                           domain=str,
                           default=None,
                           )

    def ph_args(self):
        self.add_to_config("linearize_binary_proximal_terms",
                              description="For PH, linearize the proximal terms for "
                              "all binary nonanticipative variables",
                              domain=bool,
                              default=False)


        self.add_to_config("linearize_proximal_terms",
                              description="For PH, linearize the proximal terms for "
                              "all nonanticipative variables",
                              domain=bool,
                              default=False)


        self.add_to_config("proximal_linearization_tolerance",
                            description="For PH, when linearizing proximal terms, "
                            "a cut will be added if the proximal term approximation "
                            "is looser than this value (default 1e-1)",
                            domain=float,
                            default=1.e-1)

        self.add_to_config("smoothing",
                           description="For PH, add a smoothing term to the objective",
                           domain=bool,
                           default=False)

        self.add_to_config("smoothing_rho_ratio",
                           description="For PH, when smoothing, the ratio of "
                           "the smoothing coefficient to rho (default 1e-1)",
                           domain=float,
                           default=1.e-1)

        self.add_to_config("smoothing_beta",
                           description="For PH, when smoothing, the smoothing "
                           "memory coefficient beta (default 2e-1)",
                           domain=float,
                           default=2.e-1)

    def make_parser(self, progname=None, num_scens_reqd=False):
        raise RuntimeError("make_parser is no longer used. See comments at top of config.py")


    def num_scens_optional(self):
        self.add_to_config(
            "num_scens",
            description="Number of scenarios (default None)",
            domain=int,
            default=None,
        )

    def num_scens_required(self):
        # required, but not postional
        self.add_to_config(
            "num_scens",
            description="Number of scenarios (default None)",
            domain=int,
            default=None,
            argparse_args = {"required": True}
        )


    def _basic_multistage(self, progname=None, num_scens_reqd=False):
        raise RuntimeError("_basic_multistage is no longer used. See comments at top of config.py")

    def add_branching_factors(self):
        if "branching_factors" in self:
            return
        self.add_to_config("branching_factors",
                            description="Spaces delimited branching factors (e.g., 2 2)",
                            domain=pyofig.ListOf(int, pyofig.PositiveInt),
                            default=None)
    
    def add_stage2_ef_solver_name_arg(self):
        if "stage2_ef_solver_name" in self:
            return
        self.add_to_config("stage2_ef_solver_name",
                           description="Solver for stage 2 EF in multistage xhat (default None)",
                           domain=str,
                           default=None)

    def make_multistage_parser(self, progname=None):
        raise RuntimeError("make_multistage_parser is no longer used. See comments at top of config.py")

    def multistage(self):
        self.add_branching_factors()
        self.popular_args()
        self.add_stage2_ef_solver_name_arg()


    #### EF ####
    def EF_base(self):
        self.add_solver_specs(prefix="EF")

        self.add_to_config("EF_mipgap",
                           description="mip gap option for the solver if needed (default None)",
                           domain=float,
                           default=None)
        self.add_to_config("tee_EF",
                           description="Show log output if solving the extensive form directly",
                           domain=bool,
                           default=False)

        # Some EF programs only do EF and don't check this.
        self.add_to_config("EF",
                           description="Solve the extensive form directly; ignore most other directives.",
                           domain=bool,
                           default=False)


    def EF2(self):
        self.EF_base()
        self.add_to_config("num_scens",
                           description="Number of scenarios (default None)",
                           domain=int,
                           default=None)


    def EF_multistage(self):
        self.EF_base()
        # branching factors???

    #### chance constraints (PySP-style SAA; EF only) ####
    def chance_constraint_args(self):
        # The user supplies a per-scenario binary indicator (z_s == 1 means the
        # risky constraint is satisfied in scenario s) and the big-M link; we add
        # the aggregator Sum_s p_s z_s >= 1 - cc_alpha to the EF.  See
        # doc/designs/chance_constraint_design.md.  EF only: the aggregator
        # couples all scenarios, so it does not separate for decomposition.
        self.add_to_config("cc_indicator_var",
                           description="Name of the per-scenario binary indicator "
                                       "variable for a chance constraint (z==1 means "
                                       "the risky constraint is satisfied). Enables "
                                       "the chance constraint; EF only.",
                           domain=str,
                           default=None)
        self.add_to_config("cc_alpha",
                           description="Allowed violation probability for the chance "
                                       "constraint, 0 <= alpha < 1 (alpha=0 forces "
                                       "satisfaction in every scenario). Default 0.0.",
                           domain=float,
                           default=0.0)

    ##### Lshaped #####        

    def lshaped_args(self):

        self.add_to_config(name="lshaped_hub",
                           description="Use LShaped Hub (default False)",
                           domain=bool,
                           default=False)        
        

    ##### common additions to the command line #####

    def two_sided_args(self):
        # add commands to  and also return the result

        self.add_to_config("rel_gap",
                            description="relative termination gap (default 0.05)",
                            domain=float,
                            default=0.05)

        self.add_to_config("abs_gap",
                            description="absolute termination gap (default 0)",
                            domain=float,
                            default=0.)

        self.add_to_config("max_stalled_iters",
                            description="maximum iterations with no reduction in gap (default 100)",
                            domain=int,
                            default=100)

    def timed_mipgap_args(self):
        self.add_to_config('timed_mipgap',
                           description="use a time-dependent target mip gap",
                           domain=bool,
                           default=False)

        self.add_to_config("timed_mipgap_options",
                           description=
                           "Should be a string with the following format: 'gap1:time1 gap2:time2 ... gapN:timeN'."
                           "Each pair defines a soft solver time limit, i.e. time limit only applied to solver "
                           "if MIP gap is below specified threshold. Default: 0.05:600",
                           domain=str,
                           default="0.05:600")

    def mip_options(self):
        self.add_mipgap_specs()

    def aph_args(self):
        
        self.add_to_config(name="APH",
                           description="Use APH instead of PH (default False)",
                           domain=bool,
                           default=False)
        self.add_to_config('aph_gamma',
                            description='Gamma parameter associated with asychronous projective hedging (default 1.0)',
                            domain=float,
                            default=1.0)
        self.add_to_config('aph_nu',
                            description='Nu parameter associated with asychronous projective hedging (default 1.0)',
                            domain=float,
                            default=1.0)
        self.add_to_config('aph_frac_needed',
                            description='Fraction of sub-problems required before computing projective step (default 1.0)',
                            domain=float,
                            default=1.0)
        self.add_to_config('aph_dispatch_frac',
                            description='Fraction of sub-problems to dispatch at each step of asychronous projective hedging (default 1.0)',
                            domain=float,
                            default=1.0)
        self.add_to_config('aph_sleep_seconds',
                            description='Spin-lock sleep time for APH (default 0.01)',
                            domain=float,
                            default=0.01)

    def subgradient_args(self):

        self.add_to_config(name="subgradient_hub",
                           description="Use subgradient hub instead of PH (default False)",
                           domain=bool,
                           default=False)

    def ph_primal_args(self):

        self.add_to_config(name="ph_primal_hub",
                           description="Use PH Hub which only supplies nonants (and not Ws) (default False)",
                           domain=bool,
                           default=False)

    def fixer_args(self):

        self.add_to_config('fixer',
                           description="have an integer fixer extension",
                           domain=bool,
                           default=False)

        self.add_to_config("fixer_tol",
                           description="fixer bounds tolerance  (default 1e-4)",
                           domain=float,
                           default=1e-4)

    def cvar_args(self):

        self.add_to_config('cvar',
                           description="apply the CVaR (Conditional Value-at-Risk) "
                                       "risk-management transform to every scenario "
                                       "(default False)",
                           domain=bool,
                           default=False)

        self.add_to_config("cvar_weight",
                           description="beta >= 0, the weight on CVaR in "
                                       "lambda*E[Cost] + beta*CVaR (default 1.0)",
                           domain=float,
                           default=1.0)

        self.add_to_config("cvar_alpha",
                           description="CVaR confidence level alpha, 0 < alpha < 1 "
                                       "(default 0.95)",
                           domain=float,
                           default=0.95)

        self.add_to_config("cvar_mean_weight",
                           description="lambda >= 0, the weight on E[Cost] in "
                                       "lambda*E[Cost] + beta*CVaR; use 0 for pure "
                                       "CVaR (default 1.0)",
                           domain=float,
                           default=1.0)

    def relaxed_ph_fixer_args(self):

        self.add_to_config('relaxed_ph_fixer',
                           description="have a relaxed PH fixer extension ",
                           domain=bool,
                           default=False)

        self.add_to_config("relaxed_ph_fixer_tol",
                           description="relaxed PH fixer bounds tolerance  (default 1e-4)",
                           domain=float,
                           default=1e-4)

    def integer_relax_then_enforce_args(self):
        self.add_to_config('integer_relax_then_enforce',
                           description="have an integer relax then enforce extensions",
                           domain=bool,
                           default=False)

        self.add_to_config('integer_relax_then_enforce_ratio',
                           description="fraction of time limit or iterations (whichever is sooner) "
                                       "to spend with relaxed integers",
                           domain=float,
                           default=0.5)

    def slamming_args(self):
        # Phase-1 preference-driven slamming (see doc/designs/slamming_design.md).
        # The Slammer extension is activated iff slamming_directives_file is set;
        # supplying the other slam options without the file is a hard error
        # (enforced in checker()) so that a run with no slamming options behaves
        # exactly as it does today.
        self.add_to_config("slamming_directives_file",
                           description="CSV of by-name (wildcard) slamming "
                           "directives; its presence activates the Slammer "
                           "extension (default None)",
                           domain=str,
                           default=None)

        self.add_to_config("slam_start_iter",
                           description="first hub iteration at which slamming "
                           "may occur (default 1); requires "
                           "--slamming-directives-file",
                           domain=int,
                           default=None)

        self.add_to_config("iters_between_slams",
                           description="once started, slam at most once every "
                           "this many iterations (default 1); requires "
                           "--slamming-directives-file",
                           domain=int,
                           default=None)

    def w_oscillation_args(self):
        # Detect (and optionally interrupt) oscillation/cycling in the PH W
        # vector (see doc/designs/w_oscillation_design.md). The
        # WOscillationMonitor extension is activated iff either flag is set;
        # with neither a run behaves exactly as it does today. Detection alone
        # is pure observation (no algorithm change); interruption acts on the
        # cycling nonants (slamming) and implies detection.
        self.add_to_config("detect_W_oscillations",
                           description="path to a JSON control file for "
                           "W-oscillation detection; its presence activates the "
                           "WOscillationMonitor extension and CSV reporting "
                           "(default None)",
                           domain=str,
                           default=None)

        self.add_to_config("interrupt_W_oscillations",
                           description="path to a JSON control file for "
                           "W-oscillation interruption (slamming); its "
                           "presence activates the "
                           "WOscillationMonitor extension in interrupt mode, "
                           "which runs the detection engine to drive the "
                           "actions (CSV reporting stays opt-in via a 'detect' "
                           "block) (default None)",
                           domain=str,
                           default=None)

    def reduced_costs_rho_args(self):
        self.add_to_config("reduced_costs_rho",
                           description="DEPRECATED and removed (2026-06-14); "
                                       "selecting it raises an error. Reduced-cost "
                                       "rho was not effective in practice and did "
                                       "not support flexible rank assignments. "
                                       "Consider grad_rho. See "
                                       "https://github.com/Pyomo/mpi-sppy/issues/673",
                           domain=bool,
                           default=False)
        self.add_to_config("reduced_costs_rho_multiplier",
                           description="DEPRECATED (no effect); see reduced_costs_rho",
                           domain=float,
                           default=1.0)

    def sep_rho_args(self):
        self.add_to_config("sep_rho",
                           description="have an extension that computes rho using the seprho method from the Watson/Woodruff CMS paper",
                           domain=bool,
                           default=False)
        self.add_to_config("sep_rho_multiplier",
                           description="multiplier for SepRho (default 1.0)",
                           domain=float,
                           default=1.0)


    def sensi_rho_args(self):
        self.add_to_config("sensi_rho",
                           description="have an extension that sets rho values based on objective function sensitivity",
                           domain=bool,
                           default=False)
        self.add_to_config("sensi_rho_multiplier",
                           description="multiplier for SensiRho (default 1.0)",
                           domain=float,
                           default=1.0)


    def coeff_rho_args(self):
        self.add_to_config("coeff_rho",
                           description="have a CoeffRho extension",
                           domain=bool,
                           default=False)
        self.add_to_config("coeff_rho_multiplier",
                           description="multiplier for CoeffRho (default 1.0)",
                           domain=float,
                           default=1.0)


    def gapper_args(self, name=None):
        if name is None:
            name = ""
        else:
            name = name+"_"

        self.add_to_config(f'{name}mipgaps_json',
                           description="path to json file with a mipgap schedule for PH iterations "
                                       "(planned for deprecation in a future release)",
                           domain=str,
                           default=None)

        self.add_to_config(f'{name}starting_mipgap',
                           description="Sets automatic gapper mode and the starting and minimum mipgap",
                           domain=float,
                           default=None)

        self.add_to_config(f'{name}mipgap_ratio',
                           description="The ratio of the overall relative optimality gap to the subproblem "
                                       "mipgaps. This should be less than 1 for the algorithm to make progress. "
                                       "(default = 0.1)",
                           domain=float,
                           default=0.1)

    def fwph_args(self):

        self.add_to_config('fwph',
                           description="have an fwph spoke",
                           domain=bool,
                           default=False)

        self.add_to_config('fwph_rank_ratio',
                           description="MPI ranks for the fwph spoke "
                                       "relative to the hub (flexible rank "
                                       "assignments; default 1.0 = equal)",
                           domain=float,
                           default=1.0)

        self.add_to_config(name="fwph_hub",
                           description="Use FWPH hub instead of PH (default False)",
                           domain=bool,
                           default=False)

        self.add_to_config("fwph_sdm_iter_limit",
                            description="maximum fwph SDM iterations (default 1)",
                            domain=int,
                            default=1)

        self.add_to_config("fwph_weight",
                            description="fwph weight (default 0)",
                            domain=float,
                            default=0.0)

        self.add_to_config("fwph_conv_thresh",
                            description="fwph convergence threshold  (default 1e-4)",
                            domain=float,
                            default=1e-4)

        self.add_to_config("fwph_stop_check_tol",
                            description="fwph tolerance for Gamma^t (default 1e-4)",
                            domain=float,
                            default=1e-4)

        self.add_to_config("fwph_lp_start_iterations",
                            description="Number of iterations to operate on LP relaxation to warmstart fwph duals",
                            domain=int,
                            default=0)
        self.add_to_config("fwph_save_file",
                           description="If provided, passed to FWPH as options['save_file'] (cylinder rank 0 writes).",
                           domain=str,
                           default=None)
        self.add_to_config("fwph_mip_solver_name",
                           description="Solver for the FWPH MIP/LP subproblems "
                                       "(default: fall back to --solver-name). Lets you "
                                       "pair an LP/MIP-only solver (e.g. glpk, cbc) with "
                                       "a separate QP solver via --fwph-qp-solver-name.",
                           domain=str,
                           default=None)
        self.add_to_config("fwph_qp_solver_name",
                           description="Solver for the FWPH proximal QP subproblems "
                                       "(default: fall back to --solver-name). Use an "
                                       "open-source QP solver (e.g. ipopt) when "
                                       "--fwph-mip-solver-name is an LP/MIP-only solver.",
                           domain=str,
                           default=None)

        self.add_to_config(name="fwph_objgap_hub",
                           description="Use FWPH hub instead of PH (default False)",
                           domain=bool,
                           default=False)

        self.add_to_config("fwph_objgap_start_weight",
                            description="FWPH start weight -- a value of 0 starts at xbar, a value of 1 starts at the prior MIP solution (default 0)",
                            domain=float,
                            default=0.0)

        self.add_to_config("fwph_objgap_mip_fw_effort_balance",
                           description="FWPH effort balance: values closer to 1 put less pressure on the MILP solver, but require more accuracy from Frank-Wolfe. Values near 0 demand high accuracy from the MILP solver but lower accuracy from the FW procedure.",
                           domain=float,
                           default=0.5)

        self.add_to_config("fwph_objgap_decrease_base",
                           description="FWPH accuracy base: this number raised to the iteration number, times fwph_objgap_decrease_coeff, will be the accuracy required to terminate an iteration of the FW procedure. Needs to be in (0,1).",
                           domain=float,
                           default=0.9)

        self.add_to_config("fwph_objgap_decrease_coeff",
                           description="FWPH accuracy coefficient: multiplier in beta * initial_gap * (alpha ** k) for the FW convergence threshold. Needs to be greater than 0.",
                           domain=float,
                           default=3.0)

        self.add_to_config("fwph_objgap_initial_gap_floor",
                           description="FWPH accuracy initial minimum value: this number is the lowest initial value for the FW convergence criterion, in absolute terms",
                           domain=float,
                           default=1.0)
        self.add_to_config("fwph_add_cylinder_columns",
                           description="Inject columns calculated by xhat spokes into the FWPH algorithm",
                           domain=bool,
                           default=False)

    def cg_args(self):

        self.add_to_config(name="cg_hub",
                           description="Use CG hub instead of PH (default False)",
                           domain=bool,
                           default=False)
        self.add_to_config(name="sp_solver_options",
                           description="subproblem solver options",
                           domain=str,
                           default="")
        self.add_to_config(name="mp_solver_options",
                           description="master problem solver options",
                           domain=str,
                           default=" ")
        self.add_to_config(name="relaxed_nonant",
                           description="solve master problem with relaxed nonanticipativity constraint",
                           domain=bool,
                           default=False)

    def dualcg_args(self):

        self.add_to_config(name="dualcg_hub",
                           description="Use DCG hub instead of PH (default False)",
                           domain=bool,
                           default=False)

    def lagrangian_args(self):

        self.add_to_config('lagrangian',
                              description="have a lagrangian spoke",
                              domain=bool,
                              default=False)

        self.add_to_config('lagrangian_rank_ratio',
                              description="MPI ranks for the lagrangian spoke "
                                          "relative to the hub (flexible rank "
                                          "assignments; default 1.0 = equal)",
                              domain=float,
                              default=1.0)

        self.add_solver_specs("lagrangian")
        self.add_mipgap_specs("lagrangian")

        self.add_to_config('lagrangian_try_jensens_first',
                           description="before iter 0, solve the average "
                                       "scenario and send its objective as an initial "
                                       "outer bound (two-stage only; requires the "
                                       "scenario module to define average_scenario_creator; "
                                       "requires convex recourse)",
                           domain=bool,
                           default=False)


    def reduced_costs_args(self):

        self.add_to_config('reduced_costs',
                              description="have a reduced costs spoke",
                              domain=bool,
                              default=False)

        self.add_solver_specs("reduced_costs")

        self.add_to_config('rc_verbose',
                            description="verbose output for reduced costs",
                            domain=bool,
                            default=False)
        
        self.add_to_config('rc_debug',
                            description="debug output for reduced costs",
                            domain=bool,
                            default=False)
        
        self.add_to_config('rc_fixer',
                            description="use the reduced cost fixer",
                            domain=bool,
                            default=False)
        
        self.add_to_config('rc_fixer_require_improving_lagrangian',
                            description="Only consider fixing / unfixing variables after the lagrangian "
                                        "bound computed by the reduced cost spoke has improved. (default False)",
                            domain=bool,
                            default=False)

        self.add_to_config('rc_zero_tol',
                            description="vars with rc below tol will never be fixed",
                            domain=float,
                            default=1e-4)

        self.add_to_config('rc_fix_fraction_pre_iter0',
                            description="target fix fraction for rc fixer before the first iteration",
                            domain=float,
                            default=0.0)

        self.add_to_config('rc_fix_fraction_iter0',
                            description="target fix fraction for rc fixer in first iteration",
                            domain=float,
                            default=0.0)
        
        self.add_to_config('rc_fix_fraction_iterk',
                            description="target fix fraction for rc fixer in subsequent iterations",
                            domain=float,
                            default=0.0)
        
        self.add_to_config('rc_bound_tol',
                            description="tol to consider vars at bound",
                            domain=float,
                            default=1e-6)

        self.add_to_config('reduced_costs_try_jensens_first',
                           description="before iter 0, solve the average "
                                       "scenario and send its objective as an initial "
                                       "outer bound (two-stage only; requires the "
                                       "scenario module to define average_scenario_creator; "
                                       "requires convex recourse)",
                           domain=bool,
                           default=False)


    def lagranger_args(self):

        self.add_to_config('lagranger',
                            description="have a special lagranger spoke",
                              domain=bool,
                              default=False)
        self.add_mipgap_specs("lagranger")

        self.add_to_config("lagranger_rho_rescale_factors_json",
                            description="json file: rho rescale factors (default None)",
                            domain=str,
                            default=None)

        self.add_to_config('lagranger_try_jensens_first',
                           description="before iter 0, solve the average "
                                       "scenario and send its objective as an initial "
                                       "outer bound (two-stage only; requires the "
                                       "scenario module to define average_scenario_creator; "
                                       "requires convex recourse)",
                           domain=bool,
                           default=False)


    def subgradient_bounder_args(self):

        self.add_to_config('subgradient',
                              description="have a subgradient spoke",
                              domain=bool,
                              default=False)

        self.add_to_config('subgradient_rank_ratio',
                              description="MPI ranks for the subgradient spoke "
                                          "relative to the hub (flexible rank "
                                          "assignments; default 1.0 = equal)",
                              domain=float,
                              default=1.0)

        self.add_solver_specs("subgradient")
        self.add_mipgap_specs("subgradient")

        self.add_to_config("subgradient_rho_multiplier",
                           description="rescale rho (update step size) by this factor",
                           domain=float,
                           default=None)

        self.add_to_config('subgradient_try_jensens_first',
                           description="before iter 0, solve the average "
                                       "scenario and send its objective as an initial "
                                       "outer bound (two-stage only; requires the "
                                       "scenario module to define average_scenario_creator; "
                                       "requires convex recourse)",
                           domain=bool,
                           default=False)


    def ph_ob_args(self):
        raise RuntimeError("ph_ob (the --ph-ob option) and ph_ob_args were deprecated and replaced with ph_dual August 2025\n"
                           "To get the same effect as ph_ob, use --ph-dual with --ph-dual-grad-rho")

    def relaxed_ph_args(self):

        self.add_to_config("relaxed_ph",
                            description="have a relaxed PH spoke",
                            domain=bool,
                            default=False)
        self.add_to_config("relaxed_ph_rank_ratio",
                            description="MPI ranks for the relaxed_ph spoke "
                                        "relative to the hub (flexible rank "
                                        "assignments; default 1.0 = equal)",
                            domain=float,
                            default=1.0)
        self.add_to_config("relaxed_ph_rescale_rho_factor",
                            description="Used to rescale rho initially (default=1.0)",
                            domain=float,
                            default=1.0)
        self.add_solver_specs("relaxed_ph")

    def ph_xfeas_spoke_args(self):

        self.add_to_config("ph_xfeas_spoke",
                            description="have a PH xhat-feasible spoke",
                            domain=bool,
                            default=False)
        self.add_to_config("ph_xfeas_spoke_rank_ratio",
                            description="MPI ranks for the ph_xfeas spoke "
                                        "relative to the hub (flexible rank "
                                        "assignments; default 1.0 = equal)",
                            domain=float,
                            default=1.0)
        self.add_to_config("ph_xfeas_spoke_rescale_rho_factor",
                            description="Used to rescale rho initially (default=0.1)",
                            domain=float,
                            default=0.1)
        self.add_to_config("ph_xfeas_spoke_rho_multiplier",
                            description="Rescale factor for dynamic updates in ph_xfeas_spoke if ph_xfeas_spoke and a rho setter are chosen;"
                            " note that it is not cummulative (default=1.0)",
                            domain=float,
                            default=1.0)
        self.add_to_config("ph_xfeas_spoke_grad_order_stat",
                            description="Order stat for selecting rho if ph_xfeas_spoke and grad_rho are chosen;"
                            " note that this is impacted by the multiplier (default=0.0)",
                            domain=float,
                            default=0.0)

    def ph_dual_args(self):

        self.add_to_config("ph_dual",
                            description="have a dual PH spoke",
                            domain=bool,
                            default=False)

        self.add_to_config("ph_dual_rank_ratio",
                            description="MPI ranks for the ph_dual spoke "
                                        "relative to the hub (flexible rank "
                                        "assignments; default 1.0 = equal)",
                            domain=float,
                            default=1.0)

        self.add_solver_specs("ph_dual")

        self.add_to_config("ph_dual_rescale_rho_factor",
                            description="Used to rescale rho initially (default=0.1)",
                            domain=float,
                            default=0.1)
        self.add_to_config("ph_dual_rho_multiplier",
                            description="Rescale factor for dynamic updates in ph_dual if ph_dual and a rho setter are chosen;"
                            " note that it is not cummulative (default=1.0)",
                            domain=float,
                            default=1.0)
        self.add_to_config("ph_dual_grad_order_stat",
                            description="Order stat for selecting rho if ph_dual and ph_dual_grad_rho are chosen;"
                            " note that this is impacted by the multiplier (default=0.0)",
                            domain=float,
                            default=0.0)


    def xhatlooper_args(self):

        self.add_to_config('xhatlooper',
                              description="have an xhatlooper spoke",
                              domain=bool,
                              default=False)

        self.add_to_config("xhat_scen_limit",
                            description="scenario limit xhat looper to try (default 3)",
                            domain=int,
                            default=3)

        self.add_to_config('xhatlooper_try_jensens_first',
                           description="before entering the xhatlooper main loop, solve "
                                       "the average scenario and try its first-stage "
                                       "solution as a candidate xhat (two-stage only; "
                                       "requires the scenario module to define "
                                       "average_scenario_creator)",
                           domain=bool,
                           default=False)

        self.add_to_config('xhatlooper_try_feasible_xhat_first',
                           description="before entering the xhatlooper main loop, take "
                                       "the candidate first-stage from the scenario "
                                       "module's feasible_xhat_creator (looked up on the "
                                       "module or in <module>_auxiliary) and try it as "
                                       "an xhat. Two-stage only. Mutually exclusive "
                                       "with --xhatlooper-try-jensens-first.",
                           domain=bool,
                           default=False)

    def xhatshuffle_args(self):

        self.add_to_config('xhatshuffle',
                           description="have an xhatshuffle spoke",
                           domain=bool,
                           default=False)

        self.add_to_config('xhatshuffle_rank_ratio',
                           description="MPI ranks for the xhatshuffle spoke "
                                       "relative to the hub (flexible rank "
                                       "assignments; default 1.0 = equal)",
                           domain=float,
                           default=1.0)

        self.add_to_config('add_reversed_shuffle',
                           description="using also the reversed shuffling (multistage only, default True)",
                           domain=bool,
                           default=False)

        self.add_to_config('xhatshuffle_iter_step',
                           description="step in shuffled list between 2 scenarios to try (default None)",
                           domain=int,
                           default=None)

        self.add_to_config('xhatshuffle_try_jensens_first',
                           description="before entering the xhatshuffle main loop, solve "
                                       "the average scenario and try its first-stage "
                                       "solution as a candidate xhat (two-stage only; "
                                       "requires the scenario module to define "
                                       "average_scenario_creator)",
                           domain=bool,
                           default=False)

        self.add_to_config('xhatshuffle_try_feasible_xhat_first',
                           description="before entering the xhatshuffle main loop, take "
                                       "the candidate first-stage from the scenario "
                                       "module's feasible_xhat_creator (looked up on the "
                                       "module or in <module>_auxiliary) and try it as "
                                       "an xhat. Two-stage only. Mutually exclusive "
                                       "with --xhatshuffle-try-jensens-first.",
                           domain=bool,
                           default=False)

        self.add_stage2_ef_solver_name_arg()


    def mult_rho_args(self):

        self.add_to_config('mult_rho',
                              description="Have mult_rho extension (default False)",
                              domain=bool,
                              default=False)

        self.add_to_config('mult_rho_convergence_tolerance',
                            description="rhomult does nothing with convergence below this (default 1e-4)",
                              domain=float,
                              default=1e-4)

        self.add_to_config('mult_rho_update_stop_iteration',
                            description="stop doing rhomult rho updates after this iteration (default None)",
                            domain=int,
                            default=None)

        self.add_to_config('mult_rho_update_start_iteration',
                            description="start doing rhomult rho updates on this iteration (default 2)",
                            domain=int,
                            default=2)

    def mult_rho_to_dict(self):
        assert hasattr(self, "mult_rho")
        return {"mult_rho": self.mult_rho,
                "convergence_tolerance": self.mult_rho_convergence_tolerance,
                "rho_update_stop_iteration": self.mult_rho_update_stop_iteration,
                "rho_update_start_iteration": self.mult_rho_update_start_iteration,
                "verbose": False}



    def xhatspecific_args(self):
        # we will not try to get the specification from the command line

        self.add_to_config('xhatspecific',
                              description="have an xhatspecific spoke",
                              domain=bool,
                              default=False)

        self.add_to_config('xhatspecific_try_jensens_first',
                           description="before entering the xhatspecific main loop, solve "
                                       "the average scenario and try its first-stage "
                                       "solution as a candidate xhat (two-stage only; "
                                       "requires the scenario module to define "
                                       "average_scenario_creator)",
                           domain=bool,
                           default=False)

        self.add_to_config('xhatspecific_try_feasible_xhat_first',
                           description="before entering the xhatspecific main loop, take "
                                       "the candidate first-stage from the scenario "
                                       "module's feasible_xhat_creator (looked up on the "
                                       "module or in <module>_auxiliary) and try it as "
                                       "an xhat. Two-stage only. Mutually exclusive "
                                       "with --xhatspecific-try-jensens-first.",
                           domain=bool,
                           default=False)


    def xhatxbar_args(self):

        self.add_to_config('xhatxbar',
                              description="have an xhatxbar spoke",
                              domain=bool,
                              default=False)

        self.add_to_config('xhatxbar_rank_ratio',
                              description="MPI ranks for the xhatxbar spoke "
                                          "relative to the hub (flexible rank "
                                          "assignments; default 1.0 = equal)",
                              domain=float,
                              default=1.0)

        self.add_to_config('xhatxbar_try_jensens_first',
                           description="before entering the xhatxbar main loop, solve "
                                       "the average scenario and try its first-stage "
                                       "solution as a candidate xhat (two-stage only; "
                                       "requires the scenario module to define "
                                       "average_scenario_creator)",
                           domain=bool,
                           default=False)

        self.add_to_config('xhatxbar_try_feasible_xhat_first',
                           description="before entering the xhatxbar main loop, take "
                                       "the candidate first-stage from the scenario "
                                       "module's feasible_xhat_creator (looked up on the "
                                       "module or in <module>_auxiliary) and try it as "
                                       "an xhat. Two-stage only. Mutually exclusive "
                                       "with --xhatxbar-try-jensens-first.",
                           domain=bool,
                           default=False)


    def xhatlshaped_args(self):
        # we will not try to get the specification from the command line

        self.add_to_config('xhatlshaped',
                              description="have an xhatlshaped spoke",
                              domain=bool,
                              default=False)

    def xhat_from_file_args(self):
        # Supply an initial xhat candidate from a file. Every xhat spoke
        # (xhatlooper, xhatshufflelooper, xhatspecific, xhatxbar) that
        # descends from XhatInnerBoundBase will evaluate it once, before
        # its normal exploration loop. A .csv (node_name, variable_name,
        # value; see sputils.write_nonant_tree_csv) works for any number
        # of stages and is matched by name; a .npy holds a bare ROOT
        # vector and is two-stage only. See doc/src/xhat_from_file.rst.
        self.add_to_config("xhat_from_file",
                           description="Path to a file holding an initial xhat "
                                       "to evaluate before normal xhatter "
                                       "exploration. A .csv nonant tree "
                                       "(node_name, variable_name, value) works "
                                       "for any number of stages; a .npy ROOT "
                                       "vector is two-stage only. "
                                       "Default None (off).",
                           domain=str,
                           default=None)

    def write_xhat_file_args(self):
        # Write the incumbent xhat (the whole nonant tree) to a single
        # by-name CSV. Works for any number of stages and identically for
        # EF and cylinders runs (both route through
        # sputils.write_nonant_tree_csv). Default None (off).
        self.add_to_config("write_xhat_file",
                           description="Path to write the incumbent xhat (the "
                                       "whole nonant tree) as a single by-name "
                                       "CSV: 'node_name, variable_name, value', "
                                       "node-local names. All stages; EF and "
                                       "cylinders. Default None (off).",
                           domain=str,
                           default=None)

    def wtracker_args(self):

        self.add_to_config('wtracker',
                              description="Use a wtracker extension",
                              domain=bool,
                              default=False)

        self.add_to_config('wtracker_file_prefix',
                            description="prefix for rank by rank wtracker files (default '')",
                            domain=str,
                            default='')

        self.add_to_config('wtracker_wlen',
                            description="max length of iteration window for xtracker (default 20)",
                            domain=int,
                            default=20)

        self.add_to_config('wtracker_reportlen',
                            description="max length of long reports for xtracker (default 100)",
                            domain=int,
                            default=100)

        self.add_to_config('wtracker_stdevthresh',
                            description="Ignore moving std dev below this value (default None)",
                            domain=float,
                            default=None)


    def slammax_args(self):
        # we will not try to get the specification from the command line

        self.add_to_config('slammax',
                            description="have a slammax spoke",
                              domain=bool,
                              default=False)


    def slammin_args(self):
        # we will not try to get the specification from the command line

        self.add_to_config('slammin',
                            description="have a slammin spoke",
                              domain=bool,
                              default=False)


    def cross_scenario_cuts_args(self):
        # we will not try to get the specification from the command line

        self.add_to_config('cross_scenario_cuts',
                              description="have a cross scenario cuts spoke",
                              domain=bool,
                              default=False)

        self.add_to_config("cross_scenario_iter_cnt",
                              description="cross scen check bound improve iterations "
                              "(default 4)",
                              domain=int,
                              default=4)

        self.add_to_config("eta_bounds_mipgap",
                              description="mipgap for determining eta bounds for cross "
                              "scenario cuts (default 0.01)",
                              domain=float,
                              default=0.01)
        
    # note: grad_rho_args was subsumed by gradient_args

    def gradient_args(self):

        self.add_to_config("grad_rho_multiplier",
                           description="multiplier for GradRho (default 1.0)",
                           domain=float,
                           default=1.0)

        self.add_to_config("eval_at_xhat",
                           description="evaluate the gradient at xhat whenever available (default False)",
                           domain=bool,
                           default=False)
        self.add_to_config("indep_denom",
                           description="evaluate rho using scenario independent denominator (default False)",
                           domain=bool,
                           default=False)

        self.add_to_config('grad_rho',
                           description="use a gradient-based rho setter",
                           domain=bool,
                           default=False)

        self.add_to_config("grad_order_stat",
                           description="order statistic for rho: must be between 0 (the min) and 1 (the max); 0.5 is the average (default 0.5)",
                           domain=float,
                           default=0.5)

        self.add_to_config("grad_rho_relative_bound",
                           description="factor that bounds rho/cost",
                           domain=float,
                           default=1e2)

    def dynamic_rho_args(self): # AKA adaptive

        self.gradient_args()

        self.add_to_config('dynamic_rho_primal_crit',
                           description="Use dynamic primal criterion for some types of rho updates",
                           domain=bool,
                           default=False)

        self.add_to_config('dynamic_rho_dual_crit',
                           description="Use dynamic dual criterion for some stypes of rho updates",
                           domain=bool,
                           default=False)

        self.add_to_config("dynamic_rho_primal_thresh",
                           description="primal threshold for diff during dynamic rho calcs",
                           domain=float,
                           default=0.1)
        
        self.add_to_config("dynamic_rho_dual_thresh",
                           description="dual threshold for dirr during dynamic rho calcs",
                           domain=float,
                           default=0.1)        

    def primal_dual_rho_args(self):
        self.add_to_config("use_primal_dual_rho_updater",
                         description="Use the primal dual rho updater",
                         domain=bool,
                         default=False)
        self.add_to_config("primal_dual_rho_update_threshold",
                         description="Update threshold for when primal and dual residuals are imbalanced (default=2.0)",
                         domain=float,
                         default=2.0)
        self.add_to_config("primal_dual_rho_primal_bias",
                         description="Bias to add towards primal convergence when computing updates. "
                         "Values >1 are biased toward primal and values <1 are biased towards dual (default=1.0).",
                         domain=float,
                         default=1.0)

    def norm_rho_args(self):
        self.add_to_config("use_norm_rho_updater",
                         description="Use the norm rho updater",
                         domain=bool,
                         default=False)

    def converger_args(self):
        self.add_to_config("use_norm_rho_converger",
                         description="Use the norm rho converger",
                         domain=bool,
                         default=False)
        self.add_to_config("primal_dual_converger",
                            description="Use the primal dual converger",
                            domain=bool,
                            default=False)
        self.add_to_config("primal_dual_converger_tol",
                            description="Tolerance for primal dual converger (default 1e-2)",
                            domain=float,
                            default=1e-2)

    def tracking_args(self):
        self.add_to_config("tracking_folder",
                            description="Path of results folder (default results)",
                            domain=str,
                            default="results")
        self.add_to_config("ph_track_progress",
                            description="Adds tracking extension to all"
                            " ph opt cylinders (default False). Use --track_*"
                            " to specificy what and how to track."
                            " See mpisppy.utils.cfg_vanilla.add_ph_tracking for details",
                            domain=bool,
                            default=False)
        self.add_to_config("track_convergence",
                            description="Adds convergence tracking ie"
                                " gaps and bounds (default 0)",
                            domain=int,
                            default=0)
        self.add_to_config("track_xbars",
                            description="Adds xbar tracking (default 0)",
                            domain=int,
                            default=0)
        self.add_to_config("track_duals",
                            description="Adds w tracking (default 0)",
                            domain=int,
                            default=0)
        self.add_to_config("track_nonants",
                            description="Adds nonant tracking (default 0)",
                            domain=int,
                            default=0)
        self.add_to_config('track_scen_gaps',
                            description="Adds scenario gap tracking (default 0)",
                            domain=int,
                            default=0)
        self.add_to_config("track_reduced_costs",
                            description="Adds reduced costs tracking (default 0)",
                            domain=int,
                            default=0)

    def wxbar_read_write_args(self):
        import mpisppy.utils.w_utils.wxbarreader as wxbarreader
        wxbarreader.add_options_to_config(self)

        import mpisppy.utils.w_utils.wxbarwriter as wxbarwriter
        wxbarwriter.add_options_to_config(self)

    def proper_bundle_config(self):
        self.add_to_config('pickle_bundles_dir',
                            description="Write bundles to a dill pickle files in this dir (default None)",
                            domain=str,
                            default=None)

        self.add_to_config('unpickle_bundles_dir',
                            description="Read bundles from a dill pickle files in this dir; (default None)",
                            domain=str,
                            default=None)

        self.add_to_config("scenarios_per_bundle",
                            description="Used for `proper` bundles only (might also be used when reading pickled bundles) (default None)",
                            domain=int,
                            default=None)

    def pickle_scenarios_config(self):
        # Distinct from pickle bundles
        self.add_to_config('pickle_scenarios_dir',
                            description="Write individual scenarios to a dill pickle files in this dir (default None)",
                            domain=str,
                            default=None)

        self.add_to_config('unpickle_scenarios_dir',
                            description="Read individual scenarios_per_bundle from a dill pickle files in this dir; (default None)",
                            domain=str,
                            default=None)

    def pre_pickle_args(self):
        """Options for the pre-pickle preprocessing pipeline.

        See doc/src/pickling.rst for the full description. The pipeline runs
        in this order before each scenario or bundle is pickled:
            1. SPPresolve (if presolve_before_pickle)
            2. user callback (if pre_pickle_function is set)
            3. iteration-0 solve (if iter0_before_pickle)
        """
        self.add_to_config("presolve_before_pickle",
                           description="Run the distributed presolver (FBBT, "
                           "and OBBT if --obbt is also set) once at pickle "
                           "time so the tightened bounds are baked into the "
                           "pickle. (default False)",
                           domain=bool,
                           default=False)

        self.add_to_config("pre_pickle_function",
                           description="Dotted name of a callable with "
                           "signature fn(model, cfg) to invoke on each "
                           "scenario or bundle between presolve and the "
                           "iter0 solve. (default None, no callback)",
                           domain=str,
                           default=None)

        self.add_to_config("iter0_before_pickle",
                           description="Solve each scenario or bundle once "
                           "at pickle time with its original objective "
                           "(no W, no prox -- a PH iteration 0 solve) and "
                           "store variable values plus duals/reduced costs "
                           "inside the pickle. (default False)",
                           domain=bool,
                           default=False)

        self.add_to_config("pickle_solver_name",
                           description="Solver to use for the pickle-time "
                           "iter0 solve. If None, falls back to "
                           "cfg.solver_name. (default None)",
                           domain=str,
                           default=None)

        self.add_to_config("pickle_solver_options",
                           description="Solver options string for the "
                           "pickle-time iter0 solve. If None, falls back to "
                           "cfg.solver_options. (default None)",
                           domain=str,
                           default=None)

        self.add_to_config("iter0_from_pickle",
                           description="Trust the iter0 solution baked into "
                           "the pickle by --iter0-before-pickle and skip "
                           "PHBase.Iter0's solve loop entirely. Requires "
                           "every local scenario / bundle to carry "
                           "_mpisppy_data.pickle_metadata['iter0_before_pickle']"
                           " == True; otherwise PHBase will hard-fail. "
                           "(default False)",
                           domain=bool,
                           default=False)

    def mmw_args(self):
        self.add_to_config(
            "mmw_xhat_input_file_name",
            description="Path to .npy file with xhat for MMW confidence interval"
            " (default None; if absent and the other mmw options are given,"
            " the best xhat from the main algorithm is used)",
            domain=str,
            default=None,
        )
        self.add_to_config(
            "mmw_num_batches",
            description="Number of batches for MMW confidence interval (default None)",
            domain=int,
            default=None,
        )
        self.add_to_config(
            "mmw_batch_size",
            description="Batch size for MMW confidence interval (default None)",
            domain=int,
            default=None,
        )
        self.add_to_config(
            "mmw_start",
            description="First scenario number used by MMW (default None)",
            domain=int,
            default=None,
        )

    #================
    def create_parser(self,progname=None):
        # seldom used
        if len(self) == 0:
            raise RuntimeError("create parser called before Config is populated")
        parser = argparse.ArgumentParser(progname, conflict_handler="resolve")
        self.initialize_argparse(parser)
        return parser

    #================
    def parse_command_line(self, progname=None):
        # often used, but the return value less so
        if len(self) == 0:
            raise RuntimeError("create parser called before Config is populated")
        parser = self.create_parser(progname)
        args = parser.parse_args()

        if 'CONFIGBLOCK.config_file' in args and \
                   len(args.__dict__['CONFIGBLOCK.config_file'])>0:
                    # First read output_dir (to format certain values that may use it)
                    output_dir = args.__dict__['CONFIGBLOCK.output_dir'] if 'CONFIGBLOCK.output_dir' in args else '.'
                    # First read from config file
                    self._import_read_config_dict(self._read_config_file(args.__dict__['CONFIGBLOCK.config_file'],
                                                                         output_dir=output_dir))

        # Override values from the file with any args specified in command line
        
        args = self.import_argparse(args)
        return args

    def _read_config_file(self, path_to_configfile, output_dir='.'):
        '''
        Read values from config file and load them to a dict.
        import function (see below) is expecting the name to be 'CONFIGBLOCK.'+arg_name,
        so I'm adding that to the name.
        '''
        read_config_dict = dict()
        with open(path_to_configfile,'r') as f:
            line = f.readline()
            while line:
                line = line.split('#',1)[0].strip() # Ignore anything that may come after an '#'
                if len(line)>0:
                    l_args = line.split(':') # File lines are arg_name: arg_value. There could be ':' in arg_value
                    arg_name = 'CONFIGBLOCK.'+l_args[0].strip()
                    if len(l_args)>2:
                        arg_val = ':'.join(l_args[1:]).strip()
                    else:
                        arg_val = l_args[1].strip().format(output_dir=output_dir)
                    # TODO: fix it so that "false" is correctly read and parsed
                    if arg_val.lower() not in ('none', 'false'):
                        # None values are defaults, do not read
                        read_config_dict[arg_name]=arg_val
                line = f.readline()
        return read_config_dict

    def _import_read_config_dict(self, read_config_dict):
        '''
        Copied from pyomo.common.config.import_argparse
        We could probably do without the 'CONFIGBLOCK' thing, but I'm not sure what purpose it serves so
        I just left it there.
        '''
        for level, prefix, value, obj in self._data_collector(None, ""):
            if obj._argparse is None:
                continue
            for _args, _kwds in obj._argparse:
                if 'dest' in _kwds:
                    _dest = _kwds['dest']
                    if _dest in read_config_dict:
                        obj.set_value(read_config_dict[_dest])
                else:
                    _dest = 'CONFIGBLOCK.' + obj.name(True)
                    if _dest in read_config_dict:
                        obj.set_value(read_config_dict[_dest])
                        del read_config_dict[_dest]

        # Check if any args in the config file were not used
        if len(read_config_dict)>0:
            for k in read_config_dict:
                print(f'Parameter {k} found in the config file is not a valid config option, ignoring.')

        return read_config_dict


#=================
if __name__ == "__main__":
    # a place for ad hoc testing by developers
    config = Config()
    config.popular_args() # populates self
    config.display()
    for i,j in config.items():
        print(i, j)
    print(dir(config))
    print(config._all_slots)
    print(config._domain)
    print(f"max_iterations {config['max_iterations']}")

    # most codes do not use create_parser; they use parse_command_line instead
    parser = config.create_parser("tester")
    parser.add_argument(
            "num_scens", help="Number of scenarios", type=int,
        )

    args=parser.parse_args(['3', '--max-iterations', '99', '--solver-name', 'cplex'])

    args = config.import_argparse(args)

    config.display()

    #parser.parse_args(['--help'])
