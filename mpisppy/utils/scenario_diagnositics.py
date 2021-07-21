import pyomo.environ as pyo

def find_problematic_scenario(solver, scenario_list, scenario_creator,
        scenario_creator_kwargs=None, solve_kwargs=None, results_checker=pyo.check_optimal_termination):
    '''
    Attributes
    ----------
    solver (Pyomo Solver, str) : A Pyomo solver object or a string
    scenario_list (list) : A list of scenarios to be passed into scenario_creator
    scenario_creator (function) : A function which creates a scenario from the scenario name
    scenario_creator_kwargs (dict) : Optional additional kwargs to the scenario_creator function
    solve_kwargs (dict) : Optional kwargs to the solve method.
    results_checker (function) : Optional function which takes a Pyomo results object as its first
                                 argument and returns True if the results are "OK" and False otherwise.

    Returns (tuple)
    -------
    (scenario_name, scenario_model) of a problematic scenario or (None, None)
    '''

    if isinstance(solver, str):
        solver = pyo.SolverFactory(solver)
    if scenario_creator_kwargs is None:
        scenario_creator_kwargs = {}
    if solve_kwargs is None:
        solve_kwargs = {}
    for s in scenario_list:
        model = scenario_creator(s, **scenario_creator_kwargs)
        try:
            results = solver.solve(model, **solve_kwargs)
        except:
            return s, model
        if not results_checker(results):
            return s, model
    return None, None
