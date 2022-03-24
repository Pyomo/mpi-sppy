import mpisppy.utils.baseparsers
from parser import add_common_args


def main() -> None:
    parser = mpisppy.utils.baseparsers.make_parser(num_scens_reqd=True)
    add_common_args(parser)
    args = parser.parse_args()


if __name__ == "__main__":
    main()
