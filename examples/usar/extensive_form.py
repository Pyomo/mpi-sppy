import itertools

import pyomo.environ as pyo

from generate_data import generate_data
import mpisppy.utils.sputils
from parser import new_parser
from scenario_creator import scenario_creator


def main() -> None:
    args = new_parser().parse_args()

    solver = pyo.SolverFactory(args.solver_name)

    if args.num_scens <= 0:
        raise ValueError("Provide a positive integer for num_scens")
    scenario_names = list(map(str, range(args.num_scens)))

    data_dicts = list(
        itertools.islice(generate_data(**vars(args)), args.num_scens))

    ef = mpisppy.utils.sputils.create_EF(
        scenario_names,
        scenario_creator,
        scenario_creator_kwargs={"data_dicts": data_dicts},
    )

    solver.solve(ef, tee=True)


if __name__ == "__main__":
    main()
