import itertools

from generate_data import generate_data
import mpisppy.opt.ef
from parser import new_parser
from scenario_creator import scenario_creator


def main() -> None:
    args = new_parser().parse_args()

    if args.num_scens <= 0:
        raise ValueError("Provide a positive integer for num_scens")
    scenario_names = list(map(str, range(args.num_scens)))

    data_dicts = list(
        itertools.islice(generate_data(**vars(args)), args.num_scens))

    ef = mpisppy.opt.ef.ExtensiveForm(
        {"solver": args.solver_name},
        scenario_names,
        scenario_creator,
        scenario_creator_kwargs={"data_dicts": data_dicts},
        model_name="USAR extensive form",
    )

    ef.solve_extensive_form(tee=True)


if __name__ == "__main__":
    main()
