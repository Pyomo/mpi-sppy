from mpisppy.opt.ph import PH
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils


def build_model(yields):
    model = pyo.ConcreteModel()

    # Variables
    model.X = pyo.Var(["WHEAT", "CORN", "BEETS"], within=pyo.NonNegativeReals)
    model.Y = pyo.Var(["WHEAT", "CORN"], within=pyo.NonNegativeReals)
    model.W = pyo.Var(
        ["WHEAT", "CORN", "BEETS_FAVORABLE", "BEETS_UNFAVORABLE"],
        within=pyo.NonNegativeReals,
    )

    # Objective function
    model.PLANTING_COST = 150 * model.X["WHEAT"] + 230 * model.X["CORN"] + 260 * model.X["BEETS"]
    model.PURCHASE_COST = 238 * model.Y["WHEAT"] + 210 * model.Y["CORN"]
    model.SALES_REVENUE = (
        170 * model.W["WHEAT"] + 150 * model.W["CORN"]
        + 36 * model.W["BEETS_FAVORABLE"] + 10 * model.W["BEETS_UNFAVORABLE"]
    )
    model.OBJ = pyo.Objective(
        expr=model.PLANTING_COST + model.PURCHASE_COST - model.SALES_REVENUE,
        sense=pyo.minimize
    )

    # Constraints
    model.CONSTR= pyo.ConstraintList()

    model.CONSTR.add(pyo.summation(model.X) <= 500)
    model.CONSTR.add(
        yields[0] * model.X["WHEAT"] + model.Y["WHEAT"] - model.W["WHEAT"] >= 200
    )
    model.CONSTR.add(
        yields[1] * model.X["CORN"] + model.Y["CORN"] - model.W["CORN"] >= 240
    )
    model.CONSTR.add(
        yields[2] * model.X["BEETS"] - model.W["BEETS_FAVORABLE"] - model.W["BEETS_UNFAVORABLE"] >= 0
    )
    model.W["BEETS_FAVORABLE"].setub(6000)

    return model


def scenario_creator(scenario_name):
    if scenario_name == "good":
        yields = [3, 3.6, 24]
    elif scenario_name == "average":
        yields = [2.5, 3, 20]
    elif scenario_name == "bad":
        yields = [2, 2.4, 16]
    else:
        raise ValueError("Unrecognized scenario name")

    model = build_model(yields)
    sputils.attach_root_node(model, model.PLANTING_COST, [model.X])
    model._mpisppy_probability = 1.0 / 3
    return model


def main():
    options = {
        "solver_name": "ipopt",
        "PHIterLimit": 200,
        "defaultPHrho": 1,
        "convthresh": 1e-7,
        "verbose": False,
        "display_progress": True,
        "display_timing": True,
        "iter0_solver_options": dict(),
        "iterk_solver_options": dict(),
        "display_convergence_detail": True
    }
    all_scenario_names = ["good", "average", "bad"]
    ph = PH(
        options,
        all_scenario_names,
        scenario_creator,
    )
    ph.ph_main()
    variables = ph.gather_var_values_to_rank0()
    for (scenario_name, variable_name) in variables:
        variable_value = variables[scenario_name, variable_name]
        print(scenario_name, variable_name, variable_value)


if __name__=='__main__':
    main()