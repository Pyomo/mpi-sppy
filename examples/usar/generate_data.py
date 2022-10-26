"""Contains helper functions to randomly generate USAR problem data.

Depot and household coordinates are generated uniformly at random.
Household sizes are sampled from an approximate Poisson distribution.
The times before which households must be rescued are obtained from a
scaled Pareto distribution, which assumes that emergency preparedness
follows a power law and households can generally survive some minimum
amount of time.
"""
import itertools
import math
import random
from typing import Any, Dict, Generator, Iterable, List, Tuple, TypeVar

import numpy as np
import scipy.stats

RESCUE_PARTY_SIZE = scipy.stats.poisson(2)  # About household size distribution
EMERGENCY_SUPPLIES_STOCK = scipy.stats.pareto(1)  # A power-law distribution
MIN_SURVIVAL_MINUTES = 3 * 24 * 60  # Before supplies such as water are needed

Coordinates = Tuple[float, float]


def generate_coords(
    num_depots: int, num_households: int, seed, **kwargs
) -> Tuple[List[Coordinates], List[Coordinates]]:
    """Samples coordinates uniformly from :math:`[0, 1) × [0, 1)`.

    Notably, also seeds the random module before sampling coordinates.

    Args:
        num_depots: Number of depot coordinates generated.
        num_households: Number of household coordinates generated.
        seed: Seed for the random module.
        **kwargs: Takes any additional, unused arguments.

    Returns:
        A 2-tuple of depot and household coords, each of given lengths.
    """
    for param in ("num_depots", "num_households"):
        if eval(param) < 0:
            raise ValueError(f"Give a nonnegative value for {param}")

    random.seed(seed)

    depot_coords = \
        [(random.random(), random.random()) for _ in range(num_depots)]
    household_coords = \
        [(random.random(), random.random()) for _ in range(num_households)]

    return depot_coords, household_coords


V = TypeVar("V")


def index(
    values: Iterable[V], idx: Iterable, *other_idxs: Iterable
) -> Dict[Any, V]:
    """Indexes `values` for inclusion in a Pyomo data dict.

    Args:
        values: An iterable of values for the returned dict.
        idx: Keys to index `values`.
        *other_idxs:
            If given, the keys are instead the Cartesian product,
            :math:`\mathtt{idx} × \mathtt{other\_idxs}[0] × \cdots`.

    Returns:
        dict containing `values` keyed by index(es).

    Raises:
        ValueError: There are fewer values than the index(es) imply.
    """
    keys = idx if not other_idxs else itertools.product(idx, *other_idxs)
    keys = list(keys)
    idxed_vals = dict(zip(keys, values))
    if len(idxed_vals) < len(keys):
        raise ValueError("A value should be given for each index element")
    return idxed_vals


def generate_data(
    time_horizon: int,
    time_unit_minutes: float,
    num_depots: int,
    num_active_depots: int,
    num_households: int,
    constant_rescue_time: int,
    travel_speed: float,
    constant_depot_inflow: int,
    seed,
    **kwargs,
) -> Generator[Dict, None, None]:
    """Generates USAR scenario data dictionaries ad infinitum.

    Args:
        time_horizon: Number of time steps considered in optimization.
        time_unit_minutes: Number of minutes per time step.
        num_depots: Passed to `generate_coords`.
        num_active_depots: Number of depots that can host rescue teams.
        num_households: Passed to `generate_coords`.
        constant_rescue_time: Number of time steps a rescue takes.
        travel_speed: Team speed in units of coordinates per time step.
        constant_depot_inflow: Maximum depot departures per time step.
        seed: Seed for the random module.
        **kwargs: Takes any additional, unused arguments.

    Yields:
        A dict parameterizing a specific USAR problem.
    """
    for param in ("time_horizon", "num_depots", "num_households"):
        if eval(param) < 0:
            raise ValueError(f"Give a nonnegative value for {param}")
    for param in ("time_unit_minutes", "travel_speed"):
        if eval(param) <= 0:
            raise ValueError(f"Give a positive value for {param}")

    depot_coords, household_coords = \
        generate_coords(num_depots, num_households, seed)  # Sets seed

    def pairwise_times(coords1, coords2):
        for c1, c2 in itertools.product(coords1, coords2):
            travel_time = np.linalg.norm(np.subtract(c1, c2)) / travel_speed
            yield max(1, math.ceil(travel_time))

    from_depot_times = \
        itertools.cycle(pairwise_times(depot_coords, household_coords))
    inter_site_times = \
        itertools.cycle(pairwise_times(household_coords, household_coords))

    def sample_from(dist):
        return dist.ppf(random.random())

    while True:
        household_sizes = \
            [sample_from(RESCUE_PARTY_SIZE) for _ in range(num_households)]
        household_stocks = \
            [sample_from(EMERGENCY_SUPPLIES_STOCK) for _ in household_sizes]
        household_survivals_mins = \
            [MIN_SURVIVAL_MINUTES * stock for stock in household_stocks]

        def lives_to_be_saved():
            for t in range(time_horizon):
                for s in range(num_households):
                    survives_until_t = \
                        t * time_unit_minutes <= household_survivals_mins[s]
                    yield household_sizes[s] * survives_until_t

        times, depots, sites = \
            map(range, (time_horizon, num_depots, num_households))
        yield {
            None: {
                "time_horizon": {None: time_horizon},
                "depots": depots,
                "num_active_depots": {None: num_active_depots},
                "sites": sites,
                "lives_to_be_saved": index(lives_to_be_saved(), times, sites),
                "rescue_times": index(
                    itertools.repeat(constant_rescue_time), times, sites),
                "from_depot_travel_times": index(
                    from_depot_times, times, depots, sites),
                "inter_site_travel_times": index(
                    inter_site_times, times, sites, sites),
                "depot_inflows": index(
                    itertools.repeat(constant_depot_inflow), times),
            }
        }
