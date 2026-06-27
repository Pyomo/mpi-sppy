###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

from pyomo.common.dependencies import pandas as pd

from mpisppy import global_toc

# A rho computed by a heuristic (from objective coefficients, gradients,
# sensitivities, reduced costs, ...) that is at or below this magnitude is
# treated as effectively zero -- or, if negative, as uninformative -- and is
# replaced by the positive default rho (see resolve_rho). This is a
# numerical-zero tolerance, NOT a rho floor: a small but genuinely positive
# computed rho (above this) is kept as computed rather than being inflated to
# the default (GitHub issue #560).
RHO_ZERO_TOL = 1e-8


def resolve_rho(val, default_rho):
    """Resolve a heuristic-computed rho to a usable, strictly positive value.

    Args:
        val (float): the rho value a heuristic computed.
        default_rho: the positive default rho to fall back to when ``val`` is
            not a usable rho.

    Returns:
        (rho_value, used_default): if ``val > RHO_ZERO_TOL`` it is returned
        unchanged (small positive values are respected); otherwise (near-zero
        or negative -- the heuristic is uninformative) ``default_rho`` is
        returned and ``used_default`` is True.
    """
    if val > RHO_ZERO_TOL:
        return val, False
    return default_rho, True


def check_rhos_positive(opt, source=""):
    """Raise a RuntimeError if any rho value is not strictly positive.

    Progressive Hedging requires rho > 0 for every non-anticipative variable:
    rho multiplies the proximal (consensus) term ``(rho/2)*(x - xbar)**2``, so a
    non-positive value silently disables (or, if negative, inverts) enforcement
    of non-anticipativity for that variable. This is the single, central check
    used to enforce that invariant consistently (GitHub issue #560).

    Args:
        opt: an SPBase/PHBase-derived object with ``local_scenarios``.
        source (str): optional label for the call site, included in the error
            message to aid debugging (e.g. "default_rho", "rho_setter").

    Raises:
        RuntimeError: naming the scenario and variable of the first offending
            rho value found.
    """
    for sname, s in opt.local_scenarios.items():
        for ndn_i, rho in s._mpisppy_model.rho.items():
            val = rho._value
            if val is None or val <= 0:
                vname = s._mpisppy_data.nonant_indices[ndn_i].name
                where = f" ({source})" if source else ""
                raise RuntimeError(
                    f"Non-positive rho detected{where}: scenario={sname}, "
                    f"variable={vname}, rho={val}. Progressive Hedging requires "
                    "rho > 0 for every non-anticipative variable (the rho "
                    "multiplies the proximal term). If this variable has a zero "
                    "objective coefficient, supply a positive default rho rather "
                    "than allowing the value to fall to zero."
                )


def report_zero_rho_fallback(opt, source, count, default_rho, state,
                             reason="a zero objective coefficient"):
    """Emit a rank-0 summary when rho falls back to the default for some nonants.

    Rho-setting heuristics that derive rho from objective coefficients (or
    gradients) cannot produce a meaningful magnitude when the coefficient is
    zero; the formula yields 0, which is not an acceptable rho. The historical
    behavior was to silently leave (or floor) such rhos at the default. Per
    GitHub issue #560 we keep a positive fallback but report it rather than
    doing so silently.

    To avoid flooding the log with one message per variable per iteration, this
    reports an aggregate count and only re-reports when that count changes
    (caller holds ``state``). For the common case -- zero *objective
    coefficients*, which are static for a run -- this collapses to a single
    message.

    Args:
        opt: object with ``cylinder_rank``.
        source (str): label for the message (e.g. the extension class name).
        count (int): number of nonants that fell back to the default rho.
        default_rho: the positive default value used for the fallback.
        state (dict): caller-held dict used to remember the last reported count
            (keyed by ``source``); pass the same dict on every call.
        reason (str): short phrase describing why the fallback happened.
    """
    if state.get(source) == count:
        return
    state[source] = count
    if count > 0:
        global_toc(
            f"{source}: {count} nonant(s) had {reason}; rho left at the "
            f"default ({default_rho}) for them",
            opt.cylinder_rank == 0,
        )


def assign_rho_with_fallback(opt, default_rho, source, report_state,
                             value_fn, reason="a zero objective coefficient"):
    """Set rho for every nonant from a heuristic, applying the zero-rho fallback.

    This is the shared body of the rho-setting extensions (CoeffRho, SepRho,
    SensiRho, GradRho): they differ only in how the raw rho is computed. For
    every nonant in every local scenario, ``value_fn(s, ndn_i)`` supplies that
    heuristic's raw value; ``resolve_rho`` keeps it when usably positive,
    otherwise falls back to ``default_rho``. The count of distinct nonants that
    fell back is reported via ``report_zero_rho_fallback`` (see issue #560).

    Args:
        opt: PHBase-like object with ``local_scenarios`` and ``cylinder_rank``.
        default_rho: the positive fallback rho.
        source (str): label for the fallback report (e.g. the extension name).
        report_state (dict): caller-held dict used to de-duplicate the report;
            pass the same dict on every call.
        value_fn: callable ``(s, ndn_i) -> float`` giving the raw computed rho.
        reason (str): short phrase for the report describing the fallback cause.
    """
    defaulted = set()
    for s in opt.local_scenarios.values():
        for ndn_i, rho in s._mpisppy_model.rho.items():
            rho._value, used_default = resolve_rho(value_fn(s, ndn_i), default_rho)
            if used_default:
                defaulted.add(ndn_i)
    report_zero_rho_fallback(opt, source, len(defaulted), default_rho,
                             report_state, reason=reason)


def rhos_to_csv(s, filename):
    """ write the rho values to a csv "fullname", rho
    Args:
        s (ConcreteModel): the scenario Pyomo model
        filenaame (str): file to which to write
    """
    with open(filename, "w") as f:
        f.write("fullname,rho\n")
        for ndn_i, rho in s._mpisppy_model.rho.items():
            vdata = s._mpisppy_data.nonant_indices[ndn_i]
            fullname = vdata.name
            f.write(f'"{fullname}",{rho._value}\n')

            
def rho_list_from_csv(s, filename):
    """ read rho values from a file and return a list suitable for rho_setter

    The csv may use a "fullname" or "varname" column for the variable name.
    Parentheses in names are normalized to underscores so names written in the
    lp/mps label form (e.g. ``X(1)``) match the Pyomo components built by the
    reader (``X_1_``); for ordinary Pyomo names this normalization is a no-op.

    Args:
        s (ConcreteModel): scenario whence the id values come
        filename (str): name of the csv file to read ((fullname|varname), rho)
    Returns:
        retlist (list of (id, rho) tuples); list suitable for rho_setter
   """
    rhodf = pd.read_csv(filename)
    namecol = "varname" if "varname" in rhodf.columns else "fullname"
    retlist = list()
    for idx, row in rhodf.iterrows():
        fullname = str(row[namecol]).replace('(', '_').replace(')', '_')
        vo = s.find_component(fullname)
        if vo is not None:
            retlist.append((id(vo), row["rho"]))
        else:
            raise RuntimeError(f"rho values from {filename} found Var {fullname} "
                               f"that is not found in the scenario given (name={s._name})")
    return retlist
