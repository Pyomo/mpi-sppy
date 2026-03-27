###############################################################################
# Disaster relief example for TutORials 2026
###############################################################################

import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import json
from mpisppy.opt.ef import ExtensiveForm


def scenario_creator(scenario_name,
                     network_data=dict(),
                     candidates_data=dict(),
                     num_scens=None):
    m = pyo.ConcreteModel(scenario_name)

    with open(f"{scenario_name}.json",'r') as f:
        scenario_data = json.load(f)

    # -----------------
    # Sets
    # -----------------
    nodes = list(network_data["nodes"].keys())

    F_exist = list(network_data.get("facilities", {}).keys())
    R_exist = list(network_data.get("roads", {}).keys())

    F_cand = list(candidates_data.get("facilities", {}).keys())
    R_cand = list(candidates_data.get("roads", {}).keys())

    F_all = F_exist + F_cand
    R_all = R_exist + R_cand

    m.N = pyo.Set(initialize=nodes)
    m.F_exist = pyo.Set(initialize=F_exist)
    m.R_exist = pyo.Set(initialize=R_exist)
    m.F_cand = pyo.Set(initialize=F_cand)
    m.R_cand = pyo.Set(initialize=R_cand)
    m.F_all = pyo.Set(initialize=F_all)
    m.R_all = pyo.Set(initialize=R_all)

    # -----------------
    # Scenario demand aggregated by node
    # -----------------
    d_by_node = {n: 0.0 for n in nodes}
    for did, rec in scenario_data.get("demands", {}).items():
        d_by_node[str(rec["node"])] += float(rec["demand"])

    m.d = pyo.Param(m.N, initialize=lambda m, n: d_by_node[n], within=pyo.NonNegativeReals)

    # -----------------
    # Parameters: facilities
    # -----------------
    def fac_node(m, f):
        if f in network_data.get("facilities", {}):
            return str(network_data["facilities"][f]["node"])
        return str(candidates_data["facilities"][f]["node"])
    m.fac_node = pyo.Param(m.F_all, initialize=fac_node, within=pyo.Any)

    def fac_cap(m, f):
        if f in network_data.get("facilities", {}):
            return float(network_data["facilities"][f]["capacity"])
        return float(candidates_data["facilities"][f]["capacity"])  # include capacity in candidates
    m.fac_cap = pyo.Param(m.F_all, initialize=fac_cap, within=pyo.NonNegativeReals)

    def fac_op_cost(m, f):
        # scenario override if present, else base
        if f in scenario_data.get("facilities", {}):
            return float(scenario_data["facilities"][f])
        if f in network_data.get("facilities", {}):
            return float(network_data["facilities"][f].get("oper_cost", 0.0))
        return float(candidates_data["facilities"][f].get("oper_cost", 0.0))
    m.fac_op_cost = pyo.Param(m.F_all, initialize=fac_op_cost, within=pyo.NonNegativeReals)

    m.fac_inv_cpu = pyo.Param(
        m.F_cand,
        initialize=lambda m, f: float(candidates_data["facilities"][f]["inv_cost_per_unit"]),
        within=pyo.NonNegativeReals,
    )

    # -----------------
    # Parameters: roads
    # -----------------
    def road_from(m, r):
        if r in network_data.get("roads", {}):
            return str(network_data["roads"][r]["from"])
        return str(candidates_data["roads"][r]["from_node"])
    def road_to(m, r):
        if r in network_data.get("roads", {}):
            return str(network_data["roads"][r]["to"])
        return str(candidates_data["roads"][r]["to_node"])

    m.road_from = pyo.Param(m.R_all, initialize=road_from, within=pyo.Any)
    m.road_to = pyo.Param(m.R_all, initialize=road_to, within=pyo.Any)

    def road_base_cap(m, r):
        if r in network_data.get("roads", {}):
            return float(network_data["roads"][r]["capacity"])
        return float(candidates_data["roads"][r]["capacity"])  # include capacity in candidates
    m.road_base_cap = pyo.Param(m.R_all, initialize=road_base_cap, within=pyo.NonNegativeReals)

    def road_cap_scen(m, r):
        # scenario override if present, else base
        if r in scenario_data.get("roads", {}) and isinstance(scenario_data["roads"][r], dict):
            rec = scenario_data["roads"][r]
            if "capacity" in rec:
                return float(rec["capacity"])
        return float(m.road_base_cap[r])
    m.road_cap = pyo.Param(m.R_all, initialize=road_cap_scen, within=pyo.NonNegativeReals)

    def road_op_cost(m, r):
        # scenario override if present, else base
        if r in scenario_data.get("roads", {}):
            rec = scenario_data["roads"][r]
            if isinstance(rec, dict) and "oper_cost" in rec:
                return float(rec["oper_cost"])
            if isinstance(rec, (int, float)):
                return float(rec)
        if r in network_data.get("roads", {}):
            return float(network_data["roads"][r].get("oper_cost", 0.0))
        return float(candidates_data["roads"][r].get("oper_cost", 0.0))
    m.road_op_cost = pyo.Param(m.R_all, initialize=road_op_cost, within=pyo.NonNegativeReals)

    m.road_inv_cost = pyo.Param(
        m.R_cand,
        initialize=lambda m, r: float(candidates_data["roads"][r]["inv_cost"]),
        within=pyo.NonNegativeReals,
    )

    # -----------------
    # Variables
    # -----------------
    # First-stage design
    m.x_fac = pyo.Var(m.F_cand, within=pyo.Binary)
    m.x_road = pyo.Var(m.R_cand, within=pyo.Binary)

    # Second-stage operation
    m.p = pyo.Var(m.F_all, within=pyo.NonNegativeReals)   # people served at each facility
    m.flow = pyo.Var(m.R_all, within=pyo.NonNegativeReals)  # flow on each directed road

    # -----------------
    # Constraints
    # -----------------
    def facility_cap_rule(m, f):
        if f in m.F_cand:
            return m.p[f] <= m.fac_cap[f] * m.x_fac[f]
        return m.p[f] <= m.fac_cap[f]
    m.FacilityCap = pyo.Constraint(m.F_all, rule=facility_cap_rule)

    def road_cap_rule(m, r):
        if r in m.R_cand:
            return m.flow[r] <= m.road_cap[r] * m.x_road[r]
        return m.flow[r] <= m.road_cap[r]
    m.RoadCap = pyo.Constraint(m.R_all, rule=road_cap_rule)

    # Flow balance: served_at_node + outflow - inflow = demand
    def balance_rule(m, n):
        served_here = sum(m.p[f] for f in m.F_all if m.fac_node[f] == n)
        outflow = sum(m.flow[r] for r in m.R_all if m.road_from[r] == n)
        inflow = sum(m.flow[r] for r in m.R_all if m.road_to[r] == n)
        return served_here + outflow - inflow == m.d[n]
    m.FlowBalance = pyo.Constraint(m.N, rule=balance_rule)

    # -----------------
    # Objective: investment + operating
    # -----------------
    m.FirstStageCost = pyo.Expression(
        expr=sum(m.fac_inv_cpu[f] * m.fac_cap[f] * m.x_fac[f] for f in m.F_cand)
            + sum(m.road_inv_cost[r] * m.x_road[r] for r in m.R_cand)
    )

    m.SecondStageCost = pyo.Expression(
        expr=sum(m.fac_op_cost[f] * m.p[f] for f in m.F_all)
            + sum(m.road_op_cost[r] * m.flow[r] for r in m.R_all)
    )

    m.TotalCost = pyo.Objective(expr = m.FirstStageCost + m.SecondStageCost,
                                sense=pyo.minimize)

    # mpisppy: root node with first-stage vars and first-stage cost
    sputils.attach_root_node(m, m.FirstStageCost, [m.x_fac, m.x_road])

    m._mpisppy_probability = (1 / num_scens) 
    return m


def scenario_names_creator(num_scens, start=None):
    start = 0 if start is None else start
    return [f"scen{i}" for i in range(start, start + num_scens)]


def inparser_adder(cfg):
    cfg.num_scens_required()

def kw_creator(cfg):
    with open('network_data.json','r') as f:
        network_data = json.load(f)
    with open('candidates_data.json','r') as f:
        candidates_data = json.load(f)
    
    return {"num_scens": cfg.get("num_scens", None),
            "network_data":network_data,
            "candidates_data":candidates_data
            }


def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    sca = scenario_creator_kwargs.copy()
    sca["num_scens"] = sample_branching_factors[0]
    return scenario_creator(sname, **sca)


def scenario_denouement(rank, scenario_name, scenario):
    if rank!=0:
        return
    sname = scenario_name
    sc = scenario
    print("Investment plan")
    print("Facilities built: ")
    for f in sc.F_all:
        if pyo.value(sc.x_fac[f])>0.5:
            print(f"{f}. Node: {sc.fac_node[f]}. Capacity: {sc.fac_cap[f]}")
    print("Roads built: ")
    for r in sc.R_all:
        if pyo.value(sc.x_road[r])>0.5:
            print(f"{r}. From: {sc.road_from[r]}. To: {sc.road_to[r]}. Capacity: {sc.road_cap[r]}")


def custom_writer(ef, cfg):
    if not isinstance(ef, ExtensiveForm):
        return

    sc_name,sc = next(ef.scenarios())    
    print("Investment plan")
    print("Facilities built: ")
    for f in sc.F_all:
        if pyo.value(sc.x_fac[f])>0.5:
            print(f"{f}. Node: {sc.fac_node[f]}. Capacity: {sc.fac_cap[f]}")
    print("Roads built: ")
    for r in sc.R_all:
        if pyo.value(sc.x_road[r])>0.5:
            print(f"{r}. From: {sc.road_from[r]}. To: {sc.road_to[r]}. Capacity: {sc.road_cap[r]}")
