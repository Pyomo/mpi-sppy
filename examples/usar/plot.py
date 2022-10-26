"""Functions for obtaining and plotting team schedules.

The problem of assigning departures to teams and obtaining team-specific
rescue schedules is reduced to strongly connecting subgraphs of
departures and finding an Eulerian circuit for each time step. A simpler
approach may exist, but it's not clear such an approach could generalize
to solutions with travel times of 0.

Typically, you will only directly call functions prefixed with "plot\_".
"""
import itertools
import random
from typing import (
    Any,
    Callable,
    Collection,
    Counter,
    DefaultDict,
    Deque,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    OrderedDict,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from pyomo.core.base.var import IndexedVar
import pyomo.environ as pyo


def default_colors(n: int, seed=0) -> Callable[[int], Any]:
    """Gives a randomly shuffled rainbow colormap with `n` colors."""
    colors = list(plt.cm.rainbow(np.linspace(0, 1, n)))
    random.Random(seed).shuffle(colors)
    return ListedColormap(colors)


V = TypeVar("V", bound=Hashable)


def hierholzers_alg(
    out_edges: Mapping[V, Collection[V]], start_vertex: V
) -> Tuple[Deque[Tuple[V, V]], int]:
    """Finds an Eulerian circuit relative to a start vertex.

    Implements Hierholzer's algorithm ([hierholzer1873moglichkeit]_).

    .. [hierholzer1873moglichkeit]
    .. code:: bibtex

       @article{hierholzer1873moglichkeit,
         title={{\"U}ber die M{\"o}glichkeit, einen Linienzug ohne Wiederholung und ohne Unterbrechung zu umfahren},
         author={Hierholzer, Carl and Wiener, Chr},
         journal={Mathematische Annalen},
         volume={6},
         number={1},
         pages={30--32},
         year={1873},
         publisher={Springer-Verlag}
       }

    Args:
        out_edges: Gives the vertices neighboring each vertex.
        start_vertex: A vertex considered a reference/starting point.

    Returns:
        A tuple (`eulerian_circuit`, `num_rotated`), where
        `eulerian_circuit` does not necessarily start at `start_vertex`,
        and `num_rotated` is an integer offset for `start_vertex`.

    Raises:
        ValueError: If no Eulerian circuit exists for the input.
    """
    eulerian_circuit = Deque[Tuple[V, V]]()
    num_rotated = 0
    num_edges = sum(map(len, out_edges.values()))
    end_v = start_vertex
    out_edge_iters = DefaultDict[V, Iterator[V]](lambda: iter([]))
    out_edge_iters.update({k: iter(v) for k, v in out_edges.items()})
    while len(eulerian_circuit) < num_edges:
        try:
            while True:
                eulerian_circuit.append((end_v, next(out_edge_iters[end_v])))
                end_v = eulerian_circuit[-1][1]
        except StopIteration:
            cyclic = end_v == start_vertex
            if not cyclic or num_rotated >= len(eulerian_circuit):
                raise ValueError("No Eulerian circuit starts at start_vertex")
            num_rotated += 1
            eulerian_circuit.rotate(1)
            start_vertex = end_v = eulerian_circuit[-1][1]
    return eulerian_circuit, num_rotated


def tarjans_scc_alg(
    out_edges: Mapping[V, Iterable[V]], vertices: Optional[Iterable[V]] = None
) -> Generator[Set[V], None, None]:
    """Gives the strongly connected components for a directed graph.

    Implements Tarjan's strongly connected components algorithm
    ([tarjan1972depth]_).

    .. [tarjan1972depth]
    .. code:: bibtex

       @article{tarjan1972depth,
         title={Depth-first search and linear graph algorithms},
         author={Tarjan, Robert},
         journal={SIAM journal on computing},
         volume={1},
         number={2},
         pages={146--160},
         year={1972},
         publisher={SIAM}
       }


    Args:
        out_edges: Gives the vertices neighboring each vertex.
        vertices:
            Vertex set, if given. Inferred from `out_edges` otherwise.

    Yields:
        Sets of vertices comprising strongly connected components.
    """
    if vertices is None:
        vertices = set(itertools.chain(out_edges.keys(), *out_edges.values()))

    call_stack = Deque[Tuple[V, Optional[V], Optional[Iterable[V]]]]()
    dfs_num = dict()
    scc_num = dict()
    stack_for_sccs = OrderedDict[V, None]()
    i = -1
    for v in vertices:
        if v not in dfs_num:
            call_stack.append((v, None, None))
            while call_stack:
                v1, v2, v2_iter = call_stack.pop()
                if v2 is None or v2_iter is None:
                    i = dfs_num[v1] = scc_num[v1] = i + 1
                    stack_for_sccs[v1] = None
                    v2 = v1
                    v2_iter = itertools.chain(out_edges.get(v1, []), (v1,))
                for next_v2 in v2_iter:
                    if v2 in stack_for_sccs and scc_num[v2] < scc_num[v1]:
                        scc_num[v1] = scc_num[v2]
                    v2 = next_v2
                    if v2 not in dfs_num:
                        call_stack.append((v1, v2, v2_iter))
                        call_stack.append((v2, None, None))
                        break
                else:
                    if dfs_num[v1] == scc_num[v1]:
                        scc = set()
                        while v1 not in scc:
                            scc.add(stack_for_sccs.popitem()[0])
                        yield scc


Location = Hashable
Time = int
State = Tuple[Time, Location]
Start = Tuple[State]
Transition = Tuple[State, State]
Event = Union[Start, Transition]
NumberedEvent = Tuple[int, Event]


def event_walks(numbered_events: Iterable[NumberedEvent]) -> List[List[int]]:
    """Assigns departures in `numbered_events` by team.

    This function strongly connects a subgraph for each time step (in
    chronological order) and finds an Eulerian circuit for the subgraph.
    A dummy vertex strongly connects the graph and indicates a change of
    teams when interpreting the resulting Eulerian circuit.

    Returns:
        List of team walks numbering departures as in `numbered_events`.
    """
    event_numbers = DefaultDict[Event, Deque[int]](Deque)
    events_by_time = DefaultDict[Time, Deque[Event]](Deque)
    available_walks = DefaultDict[Location, Deque[List[int]]](Deque)
    unavailable_walks = DefaultDict[Event, Deque[List[int]]](Deque)

    for num, event in numbered_events:
        event_numbers[event].append(num)
        start_t, end_t = event[0][0], event[-1][0]
        events_by_time[start_t].append(event)
        if end_t != start_t:
            events_by_time[end_t].append(event)
        if len(event) == 1:
            unavailable_walks[event].append([num])

    for t, events_at_t in sorted(events_by_time.items()):
        net_degree = Counter[State]()
        out_edges = DefaultDict[State, Deque[State]](Deque)
        dummy_vertex = (t, None)

        for event in events_at_t:
            if len(event) == 1 or event[0][0] < t:
                available_walks[event[-1][1]].append(
                    unavailable_walks[event].pop())
            elif event[0][0] == t:
                net_degree[event[0]] -= 1
                net_degree[event[-1]] += 1
                out_edges[event[0]].append(event[-1])
            else:
                raise ValueError("A given edge suggests time flows backwards")

        # Make in-degree = out-degree at each vertex
        for v, d in net_degree.items():
            for _ in range(d):
                out_edges[v].append(dummy_vertex)
            for _ in range(-d):
                out_edges[dummy_vertex].append(v)
        # Try to strongly connect the graph
        for scc in tarjans_scc_alg(out_edges):
            if dummy_vertex not in scc:
                for v in scc:
                    if len(available_walks[v]) > 0:
                        out_edges[v].append(dummy_vertex)
                        out_edges[dummy_vertex].append(v)
                        break
        eulerian_circuit, num_rotated = \
            hierholzers_alg(out_edges, dummy_vertex)

        # Incorporate Eulerian circuit into walks
        eulerian_circuit.rotate(-num_rotated)  # Need to start at dummy vertex
        for edge in eulerian_circuit:
            fro, to = edge
            if fro == dummy_vertex:
                try:
                    walk = available_walks[to[1]].popleft()
                except IndexError:
                    raise ValueError("No walk is available for Event")
            elif to == dummy_vertex:
                if fro[0] == t:
                    available_walks[fro[1]].append(walk)
            else:
                walk.append(event_numbers[edge].pop())
                if to[0] > t:
                    unavailable_walks[edge].append(walk)

    return list(itertools.chain.from_iterable(available_walks.values()))


def filter_keys_by_value(indexed_var: IndexedVar) -> Iterable:
    """Filters `indexed_var` to just what has value that is truthy."""
    def value_truthyness(key):
        var = indexed_var[key]
        try:
            val = pyo.value(var)
        except ValueError:
            return False
        return bool(round(val))

    return filter(value_truthyness, indexed_var)


def assign_departures(scen_mod: pyo.ConcreteModel) -> List[List[Transition]]:
    """Reads solution in `scen_mod` and assigns departures to teams.

    Args:
        scen_mod: Model defined as in abstract.py.

    Returns:
        List of team walks consisting of (from, to) transitions, where
        from and to are tuples of time and location.
    """
    depot_deps = [
        ((t1, d), (t1 + scen_mod.from_depot_travel_times[t1, d, s], s))
        for t1, d, s in filter_keys_by_value(scen_mod.depot_departures)
    ]
    team_starts = [(dep[1],) for dep in depot_deps]
    site_deps = [
        ((t1, s1), (t1 + scen_mod.inter_site_travel_times[t1, s1, s2], s2))
        for t1, s1, s2 in filter_keys_by_value(scen_mod.site_departures)
    ]
    walks = event_walks(enumerate(itertools.chain(team_starts, site_deps)))
    return [[(depot_deps + site_deps)[i] for i in walk] for walk in walks]


def plot_walks(
    scen_mod: pyo.ConcreteModel,
    scen_name: str,
    team_colors: Optional[Callable[[int], Any]] = None,
) -> None:
    """Plots a map of rescue team movements color-coded by team.

    Args:
        scen_mod: Model with the solution to be plotted.
        scen_name: A string name for `scen_mod`.
        team_colors: If given, returns the color for a team number.
    """
    team_walks = assign_departures(scen_mod)

    if team_colors is None:
        team_colors = default_colors(len(team_walks))

    if len(scen_mod.depot_coords) > 0:
        plt.scatter(*zip(*scen_mod.depot_coords), c="k", marker="s")
    if len(scen_mod.site_coords) > 0:
        plt.scatter(*zip(*scen_mod.site_coords), c="k", marker="o")

    arrow_start_locs, arrow_directions, arrow_colors = [], [], []

    for team, walk in enumerate(team_walks):
        from_coords_map = scen_mod.depot_coords
        for (_, fro), (t2, to) in walk:
            if t2 < scen_mod.time_horizon:
                to_coords = scen_mod.site_coords[to]
                from_coords = from_coords_map[fro]
                from_coords_map = scen_mod.site_coords
                arrow_start_locs.append(from_coords)
                arrow_directions.append(np.subtract(to_coords, from_coords))
                arrow_colors.append(team_colors(team))

    if len(arrow_start_locs) > 0:
        plt.quiver(
            *zip(*arrow_start_locs),
            *zip(*arrow_directions),
            angles="xy",
            scale_units="xy",
            scale=1,
            color=arrow_colors,
        )

    plt.title("Scenario " + scen_name + " team movements color-coded by team")
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")


def plot_gantt(
    scen_mod: pyo.ConcreteModel,
    scen_name: str,
    team_colors: Optional[Callable[[int], Any]] = None,
) -> None:
    """Plots a Gantt chart of rescue schedules color-coded by team.

    Args:
        scen_mod: Model with the solution to be plotted.
        scen_name: A string name for `scen_mod`.
        team_colors: If given, returns the color for a team number.
    """
    team_walks = assign_departures(scen_mod)

    if team_colors is None:
        team_colors = default_colors(len(team_walks))

    starts, durations, are_travel, colors = [], [], [], []

    for team, walk in enumerate(team_walks):
        for (t1, _), (t2, to) in walk:
            if t2 < scen_mod.time_horizon:
                starts.extend([t1, t2])
                durations.extend([t2 - t1, scen_mod.rescue_times[t2, to]])
                are_travel.extend([True, False])
                colors.extend([team_colors(team)] * 2)
    idx = list(range(len(durations)))[::-1]

    if len(starts) > 0:
        plt.barh(idx, durations, left=starts)

        plt.margins(0.0)

        xl = plt.xlim()

        ax_pos = plt.gca().get_position()
        fig_w, fig_h = plt.gcf().get_size_inches()
        ax_w, ax_h = ax_pos.width, ax_pos.height
        avg_bar_w = fig_w * ax_w * np.mean(durations) / (xl[1] - xl[0])
        new_fig_h = fig_h * (1 - ax_h) + avg_bar_w / 10 * len(durations)
        plt.gcf().set_size_inches(fig_w, new_fig_h)
        new_ax_ymin = fig_h * ax_pos.ymin / new_fig_h
        new_ax_h = avg_bar_w / 10 * len(durations) / new_fig_h
        plt.gca().set_position((ax_pos.xmin, new_ax_ymin, ax_w, new_ax_h))

        bg_kwargs = {"width": xl[1] - xl[0], "left": xl[0], "height": 1.0}
        idx_travel = [i for i, travel in zip(idx, are_travel) if travel]
        idx_rescue = [i for i, travel in zip(idx, are_travel) if not travel]
        plt.barh(idx_travel, **bg_kwargs, color="#eeeeee", label="Travel")
        plt.barh(idx_rescue, **bg_kwargs, color="#ffffff", label="Rescue")
        plt.legend()

        plt.barh(idx, durations, height=0.7, left=starts, color=colors)

    plt.title("Scenario " + scen_name + " rescue schedule color-coded by team")
    plt.xlabel("Time step")
    plt.yticks([])
