"""Provides an importable function for creating an `AbstractModel`."""
import pyomo.environ as pyo


def abstract_model() -> pyo.AbstractModel:
    """Returns an `AbstractModel` for urban search and rescue.

    This formulation is similar to the multistage stochastic formulation
    in [chen2012optimal]_.

    .. [chen2012optimal]
    .. code:: bibtex

       @article{chen2012optimal,
           title={Optimal team deployment in urban search and rescue},
           author={Chen, Lichun and Miller-Hooks, Elise},
           journal={Transportation Research Part B: Methodological},
           volume={46},
           number={8},
           pages={984--999},
           year={2012},
           publisher={Elsevier}
       }
    """
    ab = pyo.AbstractModel()

    ab.time_horizon = pyo.Param(within=pyo.PositiveIntegers)
    ab.times = pyo.Set(
        initialize=lambda mod: pyo.RangeSet(0, mod.time_horizon - 1))

    ab.depots = pyo.Set()
    ab.num_active_depots = pyo.Param(
        validate=lambda mod, n: 0 <= n <= len(mod.depots), within=pyo.Integers)

    ab.sites = pyo.Set()

    ab.lives_to_be_saved = pyo.Param(
        ab.times, ab.sites, within=pyo.NonNegativeIntegers)

    ab.rescue_times = pyo.Param(
        ab.times, ab.sites, within=pyo.NonNegativeIntegers)

    # For a solution without site cycles, travel times cannot be 0:
    ab.from_depot_travel_times = pyo.Param(
        ab.times, ab.depots, ab.sites, within=pyo.PositiveIntegers)
    ab.inter_site_travel_times = pyo.Param(
        ab.times, ab.sites, ab.sites, within=pyo.PositiveIntegers)

    ab.depot_inflows = pyo.Param(ab.times, domain=pyo.NonNegativeIntegers)

    ab.is_active_depot = pyo.Var(ab.depots, domain=pyo.Binary)

    # Self-loops are prohibited; stays_at_site should be used instead:
    def no_self_loops(mod, _, s1, s2):
        return (0, int(s1 != s2))
    ab.site_departures = pyo.Var(
        ab.times, ab.sites, ab.sites, domain=pyo.Binary, bounds=no_self_loops)
    ab.depot_departures = pyo.Var(
        ab.times, ab.depots, ab.sites, domain=pyo.Binary)
    ab.stays_at_site = pyo.Var(
        ab.times, ab.sites, domain=pyo.Binary)
    ab.is_time_from_arrival = pyo.Var(
        ab.times, ab.times, ab.sites, domain=pyo.Binary)

    def limit_num_active_depots(mod):
        if len(mod.is_active_depot) == 0:
            return pyo.Constraint.Feasible
        return sum(mod.is_active_depot[:]) == mod.num_active_depots
    ab.limit_num_active_depots = pyo.Constraint(rule=limit_num_active_depots)

    def depart_only_active_depots(mod, time, depot, site):
        return (
            mod.depot_departures[time, depot, site]
            <= mod.is_active_depot[depot]
        )
    ab.depart_only_active_depots = pyo.Constraint(
        ab.times, ab.depots, ab.sites, rule=depart_only_active_depots)

    def limit_depot_outflow(mod, time):
        if len(mod.depot_departures) == 0:
            return pyo.Constraint.Feasible
        return sum(mod.depot_departures[time, :, :]) <= mod.depot_inflows[time]
    ab.limit_depot_outflow = pyo.Constraint(ab.times, rule=limit_depot_outflow)

    def set_is_time_from_arrival(mod, time, time_from_arrival, site):
        num_teams_en_route = 0
        if time > 0 and time_from_arrival + 1 < mod.time_horizon:
            num_teams_en_route += \
                mod.is_time_from_arrival[time - 1, time_from_arrival + 1, site]
        for d in mod.depots:
            if mod.from_depot_travel_times[time, d, site] == time_from_arrival:
                num_teams_en_route += mod.depot_departures[time, d, site]
        for s in mod.sites:
            if mod.inter_site_travel_times[time, s, site] == time_from_arrival:
                num_teams_en_route += mod.site_departures[time, s, site]
        return (
            mod.is_time_from_arrival[time, time_from_arrival, site]
            == num_teams_en_route
        )
    ab.set_is_time_from_arrival = pyo.Constraint(
        ab.times, ab.times, ab.sites, rule=set_is_time_from_arrival)

    def flow_conservation(mod, time, site):
        inflow = mod.is_time_from_arrival[time, 0, site]
        if time > 0:
            inflow += mod.stays_at_site[time - 1, site]
        outflow = (
            sum(mod.site_departures[time, site, s] for s in mod.sites)
            + mod.stays_at_site[time, site]
        )
        return inflow == outflow
    ab.flow_conservation = pyo.Constraint(
        ab.times, ab.sites, rule=flow_conservation)

    def visit_only_once(mod, site):
        return sum(mod.is_time_from_arrival[:, 0, site]) <= 1
    ab.visit_only_once = pyo.Constraint(ab.sites, rule=visit_only_once)

    def fully_service_site(mod, time, site):
        between_0_1_if_rescue_underway = sum(
            mod.is_time_from_arrival[t, 0, site]
            for t in range(time + 1)
            if t + mod.rescue_times[t, site] > time
        ) / mod.time_horizon
        return mod.stays_at_site[time, site] >= between_0_1_if_rescue_underway
    ab.fully_service_site = pyo.Constraint(
        ab.times, ab.sites, rule=fully_service_site)

    def first_stage_cost(mod):
        return 0
    ab.first_stage_cost = pyo.Expression(rule=first_stage_cost)

    def lives_saved(mod):
        return sum(
            mod.lives_to_be_saved[t, s] * mod.is_time_from_arrival[t, 0, s]
            for t in mod.times for s in mod.sites
        )
    ab.objective = pyo.Objective(rule=lives_saved, sense=pyo.maximize)

    return ab
