###############################################################################
# aircondMulti.py
# Multi-product extension of aircond.py for aph-fw experiments.
#
# Key differences from aircond.py:
#   - num_products products, each with its own demand stream and inventory
#   - Regular-time capacity is SHARED across all products:
#       sum_p RegularProd[p] <= Capacity
#   - Overtime is per-product and unconstrained (just expensive)
#   - Product p has costs: base * (1 + p * cost_spread)
#   - BeginInventory and starting_d are divided equally across products
#   - Demand seeds: start_seed + p * PRODUCT_SEED_OFFSET + node_idx(...)
#
# New CLI args: --num-products (int, default 2), --cost-spread (float, default 0.1)
###############################################################################
import pyomo.environ as pyo
import numpy as np
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import pyomo.common.config as pyofig

# Module-level random stream (re-seeded per node per product)
aircondstream = np.random.RandomState()  # pylint: disable=no-member

# Per-product demand seed separation; large enough to avoid collisions
# across any realistic tree size.
PRODUCT_SEED_OFFSET = 100_000

# Per-bundle demand seed separation.
# When mpi-sppy builds bundles it passes sub-tree BFs (e.g. [1,2,2,2]) to
# scenario_creator.  All bundles then share the same internal BF structure,
# so node_idx values are identical across bundles.  bundle_idx
# (= global scennum // sub-tree-size) is added to the seed to give each
# bundle a unique demand realisation.
# Must satisfy: max_bundle_idx * BUNDLE_SEED_OFFSET + max_node_idx
#                 < PRODUCT_SEED_OFFSET = 100_000
# Safe for up to ~999 bundles with sub-tree node counts up to ~99.
BUNDLE_SEED_OFFSET = 100

# Do not edit these defaults!
parms = {
    "mu_dev":            (float, 0.),
    "sigma_dev":         (float, 40.),
    "start_ups":         (bool,  False),
    "StartUpCost":       (float, 300.),
    "start_seed":        (int,   1134),
    "min_d":             (float, 0.),
    "max_d":             (float, 400.),
    "starting_d":        (float, 200.),   # total; each product gets starting_d / num_products
    "BeginInventory":    (float, 200.),   # total; each product gets BeginInventory / num_products
    "InventoryCost":     (float, 0.5),
    "LastInventoryCost": (float, -0.8),
    "Capacity":          (float, 200.),   # shared regular-time capacity across all products
    "RegularProdCost":   (float, 1.),     # base; product p gets base * (1 + p * cost_spread)
    "OvertimeProdCost":  (float, 3.),     # base; product p gets base * (1 + p * cost_spread)
    "NegInventoryCost":  (float, 5.),
    "QuadShortCoeff":    (float, 0.),
    "num_products":      (int,   2),
    "cost_spread":       (float, 0.1),
}


# ----------------------------- demand generation -----------------------------

def _demands_creator(product_index, sname, sample_branching_factors,
                     root_name="ROOT", **kwargs):
    """Return (demands, nodenames) for one product along scenario sname's path.

    Each product gets an independent demand stream via a seed offset of
    product_index * PRODUCT_SEED_OFFSET.  The tree path (nodenames) is
    identical for all products; only the demand values differ.
    """
    if "start_seed" not in kwargs:
        raise RuntimeError(f"start_seed not in kwargs={kwargs}")
    start_seed   = kwargs["start_seed"]
    max_d        = kwargs.get("max_d",        parms["max_d"][1])
    min_d        = kwargs.get("min_d",        parms["min_d"][1])
    mu_dev       = kwargs.get("mu_dev",       parms["mu_dev"][1])
    sigma_dev    = kwargs.get("sigma_dev",    parms["sigma_dev"][1])
    num_products = kwargs.get("num_products", parms["num_products"][1])

    scennum = sputils.extract_num(sname)
    prod    = np.prod(sample_branching_factors)
    s       = int(scennum % prod)

    # bundle_idx distinguishes scenarios that belong to different bundles but
    # share the same position within the sub-tree BFs.  When scenario_creator
    # is called with the full BFs (no bundling), all scennum < prod, so
    # bundle_idx == 0 and the behaviour is identical to before.
    bundle_idx = int(scennum // prod)

    d         = kwargs.get("starting_d", parms["starting_d"][1]) / num_products
    demands   = [d]
    nodenames = [root_name]

    for bf in sample_branching_factors:
        assert prod % bf == 0
        prod = prod // bf
        nodenames.append(str(s // prod))
        s = s % prod

    stagelist = [int(x) for x in nodenames[1:]]
    for t in range(1, len(nodenames)):
        seed = (start_seed
                + product_index * PRODUCT_SEED_OFFSET
                + bundle_idx * BUNDLE_SEED_OFFSET
                + sputils.node_idx(stagelist[:t], sample_branching_factors))
        aircondstream.seed(seed)
        d = min(max_d, max(min_d, d + aircondstream.normal(mu_dev, sigma_dev)))
        demands.append(d)

    return demands, nodenames


# ----------------------------- rho setters -----------------------------------

def general_rho_setter(scenario_instance, rho_scale_factor=1.0):
    computed_rhos = []
    num_products  = scenario_instance.num_products
    for t in scenario_instance.T[:-1]:
        sm = scenario_instance.stage_models[t]
        for p in range(num_products):
            computed_rhos.append((id(sm.RegularProd[p]),
                                  sm.RegularProdCost[p] * rho_scale_factor))
            computed_rhos.append((id(sm.OvertimeProd[p]),
                                  sm.OvertimeProdCost[p] * rho_scale_factor))
    return computed_rhos


def dual_rho_setter(scenario_instance):
    return general_rho_setter(scenario_instance, rho_scale_factor=0.0001)


def primal_rho_setter(scenario_instance):
    return general_rho_setter(scenario_instance, rho_scale_factor=0.01)


# ----------------------------- stage model -----------------------------------

def _StageModel_creator(time, demands_by_product, last_stage, **kwargs):
    """Build the Pyomo model for a single stage of the multi-product problem.

    demands_by_product: plain Python dict {p: scalar demand at this stage}
    """
    def _kw(pname):
        return kwargs.get(pname, parms[pname][1])

    num_products = _kw("num_products")
    cost_spread  = _kw("cost_spread")

    model = pyo.ConcreteModel()
    model.T           = [time]
    model.num_products = num_products

    # Demand per product (plain Python dict, not a Pyomo Param)
    model.Demand = demands_by_product

    model.InventoryCost     = _kw("InventoryCost")
    model.LastInventoryCost = _kw("LastInventoryCost")
    model.Capacity          = _kw("Capacity")
    model.max_T             = 25
    model.bigM              = model.Capacity * model.max_T
    model.start_ups         = _kw("start_ups")
    model.NegInventoryCost  = _kw("NegInventoryCost")
    model.QuadShortCoeff    = _kw("QuadShortCoeff")

    base_reg = _kw("RegularProdCost")
    base_ot  = _kw("OvertimeProdCost")

    # Per-product costs: plain Python dicts so they survive Pyomo serialisation
    model.RegularProdCost  = {p: base_reg * (1.0 + p * cost_spread)
                              for p in range(num_products)}
    model.OvertimeProdCost = {p: base_ot  * (1.0 + p * cost_spread)
                              for p in range(num_products)}

    if model.start_ups:
        model.StartUpCost = _kw("StartUpCost")

    # --- Variables (indexed by product) ---
    model.RegularProd  = pyo.Var(range(num_products), domain=pyo.NonNegativeReals,
                                 bounds=(0, model.bigM))
    model.OvertimeProd = pyo.Var(range(num_products), domain=pyo.NonNegativeReals,
                                 bounds=(0, model.bigM))
    model.Inventory    = pyo.Var(range(num_products), domain=pyo.Reals,
                                 bounds=(-model.bigM, model.bigM))

    if model.start_ups:
        # One binary per stage: production happened at all this period
        model.StartUp = pyo.Var(within=pyo.Binary)

    # --- Constraints ---

    # Shared regular-time capacity across all products
    model.MaximumCapacity = pyo.Constraint(
        expr=sum(model.RegularProd[p] for p in range(num_products)) <= model.Capacity
    )

    if model.start_ups:
        # Any production (regular or overtime, any product) incurs the startup cost
        model.RegStartUpConstraint = pyo.Constraint(
            expr=model.bigM * model.StartUp >= sum(
                model.RegularProd[p] + model.OvertimeProd[p]
                for p in range(num_products)
            )
        )

    # Positive/negative inventory decomposition per product
    assert model.InventoryCost > 0
    assert model.NegInventoryCost > 0

    model.negInventory = pyo.Var(range(num_products), domain=pyo.NonNegativeReals,
                                 initialize=0.0, bounds=(0, model.bigM))
    model.posInventory = pyo.Var(range(num_products), domain=pyo.NonNegativeReals,
                                 initialize=0.0, bounds=(0, model.bigM))

    def dole_inventory_rule(m, p):
        return m.Inventory[p] == m.posInventory[p] - m.negInventory[p]
    model.doleInventory = pyo.Constraint(range(num_products), rule=dole_inventory_rule)

    # Inventory cost expression per product
    def inven_cost_expr_rule(m, p):
        if not last_stage:
            lin = (m.InventoryCost * m.posInventory[p]
                   + m.NegInventoryCost * m.negInventory[p])
        else:
            assert m.LastInventoryCost < 0, \
                f"last stage inven cost must be negative: {m.LastInventoryCost}"
            lin = (m.LastInventoryCost * m.posInventory[p]
                   + m.NegInventoryCost * m.negInventory[p])
        if m.QuadShortCoeff > 0 and not last_stage:
            return lin + m.QuadShortCoeff * m.negInventory[p] * m.negInventory[p]
        return lin

    model.InvenCostExpr = pyo.Expression(range(num_products),
                                         rule=inven_cost_expr_rule)

    # Stage objective: sum over products, plus one shared startup cost if enabled
    def stage_objective_rule(m):
        expr = sum(
            m.RegularProdCost[p]  * m.RegularProd[p]
            + m.OvertimeProdCost[p] * m.OvertimeProd[p]
            + m.InvenCostExpr[p]
            for p in range(num_products)
        )
        if m.start_ups:
            expr += m.StartUpCost * m.StartUp
        return expr

    model.StageObjective = pyo.Objective(rule=stage_objective_rule,
                                         sense=pyo.minimize)
    return model


# ----------------------------- full scenario model ---------------------------

def aircondMulti_model_creator(demands_by_product, **kwargs):
    """Build the complete multi-stage, multi-product Pyomo scenario model.

    demands_by_product: dict {p: [demand_t1, demand_t2, ...]}
    """
    num_products = kwargs.get("num_products", parms["num_products"][1])
    start_ups    = kwargs.get("start_ups",    parms["start_ups"][1])

    num_stages = len(demands_by_product[0])

    model = pyo.ConcreteModel()
    model.T            = range(1, num_stages + 1)
    model.start_ups    = start_ups
    model.num_products = num_products
    model.max_T        = 25

    if model.T[-1] > model.max_T:
        raise RuntimeError(f"Number of stages ({model.T[-1]}) exceeds max_T={model.max_T}")

    begin_inv_total = kwargs.get("BeginInventory", parms["BeginInventory"][1])
    # Plain Python dict — not a Pyomo Param — so material_balance_rule can
    # use m.BeginInventory[p] as a numeric value in a Pyomo expression.
    model.BeginInventory = {p: begin_inv_total / num_products
                            for p in range(num_products)}

    # Create stage sub-models
    model.stage_models = {}
    for t in model.T:
        last_stage    = (t == num_stages)
        stage_demands = {p: demands_by_product[p][t - 1] for p in range(num_products)}
        model.stage_models[t] = _StageModel_creator(
            t, stage_demands, last_stage=last_stage, **kwargs
        )

    # Per-product, per-stage material balance linking inventory across stages
    def material_balance_rule(m, t, p):
        sm = m.stage_models[t]
        if t == 1:
            return (m.BeginInventory[p]
                    + sm.RegularProd[p] + sm.OvertimeProd[p] - sm.Inventory[p]
                    == sm.Demand[p])
        else:
            return (m.stage_models[t - 1].Inventory[p]
                    + sm.RegularProd[p] + sm.OvertimeProd[p] - sm.Inventory[p]
                    == sm.Demand[p])

    model.MaterialBalance = pyo.Constraint(model.T, range(num_products),
                                           rule=material_balance_rule)

    # Deactivate per-stage objectives; build a single total-cost objective
    for t in model.T:
        model.stage_models[t].StageObjective.deactivate()

    def total_cost_rule(m):
        return sum(
            sum(
                m.stage_models[t].RegularProdCost[p]  * m.stage_models[t].RegularProd[p]
                + m.stage_models[t].OvertimeProdCost[p] * m.stage_models[t].OvertimeProd[p]
                + m.stage_models[t].InvenCostExpr[p]
                for p in range(num_products)
            ) + (m.stage_models[t].StartUpCost * m.stage_models[t].StartUp
                 if m.start_ups else 0)
            for t in m.T
        )

    model.TotalCostObjective = pyo.Objective(rule=total_cost_rule,
                                             sense=pyo.minimize)

    # Expose stage sub-models as named attributes (required by mpi-sppy)
    for t in model.T:
        setattr(model, "stage_model_" + str(t), model.stage_models[t])

    return model


# ----------------------------- scenario tree ---------------------------------

def MakeNodesforScen(model, nodenames, branching_factors, starting_stage=1):
    """Build the ScenarioNode list for a multi-product aircond scenario."""
    num_products = model.num_products
    TreeNodes    = []
    ndn          = None

    for stage in model.T:
        sm = model.stage_models[stage]

        # Nonants: RegularProd and OvertimeProd for each product, interleaved
        nonant_list = []
        for p in range(num_products):
            nonant_list.append(sm.RegularProd[p])
            nonant_list.append(sm.OvertimeProd[p])

        nonant_ef_suppl_list = [sm.Inventory[p] for p in range(num_products)]
        if model.start_ups:
            nonant_ef_suppl_list.append(sm.StartUp)

        if stage == 1:
            ndn = "ROOT"
            TreeNodes.append(scenario_tree.ScenarioNode(
                name=ndn,
                cond_prob=1.0,
                stage=stage,
                cost_expression=sm.StageObjective,
                nonant_list=nonant_list,
                scen_model=model,
                nonant_ef_suppl_list=nonant_ef_suppl_list,
            ))
        elif stage <= starting_stage:
            parent_ndn = ndn
            ndn = parent_ndn + "_0"   # single node per stage before starting_stage
            TreeNodes.append(scenario_tree.ScenarioNode(
                name=ndn,
                cond_prob=1.0,
                stage=stage,
                cost_expression=sm.StageObjective,
                nonant_list=nonant_list,
                scen_model=model,
                nonant_ef_suppl_list=nonant_ef_suppl_list,
                parent_name=parent_ndn,
            ))
        elif stage < max(model.T):
            parent_ndn = ndn
            ndn = parent_ndn + "_" + nodenames[stage - starting_stage]
            TreeNodes.append(scenario_tree.ScenarioNode(
                name=ndn,
                cond_prob=1.0 / branching_factors[stage - starting_stage - 1],
                stage=stage,
                cost_expression=sm.StageObjective,
                nonant_list=nonant_list,
                scen_model=model,
                nonant_ef_suppl_list=nonant_ef_suppl_list,
                parent_name=parent_ndn,
            ))

    return TreeNodes


# ----------------------------- mpi-sppy interface ----------------------------

def scenario_creator(sname, **kwargs):
    if "branching_factors" not in kwargs:
        raise RuntimeError(
            "scenario_creator for aircondMulti needs branching_factors in kwargs"
        )
    branching_factors = kwargs["branching_factors"]
    num_products      = kwargs.get("num_products", parms["num_products"][1])

    # Build independent demand sequences for each product
    demands_by_product = {}
    nodenames          = None
    for p in range(num_products):
        demands, nnames = _demands_creator(p, sname, branching_factors,
                                           root_name="ROOT", **kwargs)
        demands_by_product[p] = demands
        if nodenames is None:
            nodenames = nnames   # tree structure is the same for all products

    model = aircondMulti_model_creator(demands_by_product, **kwargs)
    model._mpisppy_node_list    = MakeNodesforScen(model, nodenames, branching_factors)
    model._mpisppy_probability  = "uniform"
    return model


def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                              given_scenario=None, **scenario_creator_kwargs):
    raise NotImplementedError(
        "sample_tree_scen_creator is not yet implemented for aircondMulti"
    )


def scenario_names_creator(num_scens, start=None):
    if start is None:
        start = 0
    return [f"scen{i}" for i in range(start, start + num_scens)]


def inparser_adder(cfg):
    def _doone(name, helptext):
        h = f"{helptext} (default {parms[name][1]})"
        cfg.add_to_config(name,
                          description=h,
                          domain=parms[name][0],
                          default=parms[name][1],
                          argparse=True)

    cfg.add_to_config("branching_factors",
                      description="branching factors",
                      domain=pyofig.ListOf(int),
                      default=None)
    _doone("mu_dev",            "average deviation of demand between two periods")
    _doone("sigma_dev",         "standard deviation of demand deviation between periods")
    _doone("start_ups",         "include start-up costs (results in a MIP)")
    _doone("StartUpCost",       "fixed cost when any production occurs in a period")
    _doone("start_seed",        "random number seed")
    _doone("min_d",             "minimum demand per product per period")
    _doone("max_d",             "maximum demand per product per period")
    _doone("starting_d",        "total demand at period 0, split equally across products")
    _doone("InventoryCost",     "per-period per-unit inventory holding cost")
    _doone("BeginInventory",    "total initial inventory, split equally across products")
    _doone("LastInventoryCost", "inventory value (negative) in the final period")
    _doone("Capacity",          "shared regular-time capacity across all products")
    _doone("RegularProdCost",   "base regular-time cost; product p gets base*(1+p*cost_spread)")
    _doone("OvertimeProdCost",  "base overtime cost; product p gets base*(1+p*cost_spread)")
    _doone("NegInventoryCost",  "per-unit backorder cost (positive coefficient)")
    _doone("QuadShortCoeff",    "quadratic backorder coefficient (non-negative)")
    _doone("num_products",      "number of products sharing the regular-time capacity")
    _doone("cost_spread",       "cost increment: product p costs base*(1 + p*cost_spread)")


def kw_creator(cfg, optionsin=None):
    options = optionsin if optionsin is not None else {}
    if "kwargs" in options:
        return options["kwargs"]

    kwargs = {}

    def _kwarg(option_name, default=None, arg_name=None):
        retval = options.get(option_name)
        if retval is not None:
            kwargs[option_name] = retval
            return
        aname  = option_name if arg_name is None else arg_name
        retval = cfg.get(aname)
        kwargs[option_name] = default if retval is None else retval

    _kwarg("branching_factors")
    for idx, tpl in parms.items():
        _kwarg(idx, tpl[1])

    if kwargs.get("start_ups") is None:
        raise ValueError(f"kw_creator: no value for start_ups; options={options}")
    if kwargs.get("start_seed") is None:
        raise ValueError(f"kw_creator: no value for start_seed; options={options}")

    if kwargs.get("branching_factors") is not None:
        BFs = kwargs["branching_factors"]
        ns  = cfg.get("num_scens")
        if BFs is not None and ns is None:
            cfg.add_and_assign("num_scens",
                               description="Number of scenarios",
                               domain=int,
                               value=int(np.prod(BFs)),
                               default=None)
    return kwargs


def scenario_denouement(rank, scenario_name, scenario):
    pass
