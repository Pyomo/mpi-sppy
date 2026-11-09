"""Read main_case files from pyfory_main_writer.py in a fresh process."""

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import dill
from mpisppy.utils import pyfory_model_codec as codec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    directory = parser.parse_args().directory

    with (directory / "main_case.dill").open("rb") as stream:
        model = dill.load(stream)
    assert model.callback(2) == 3
    print("dill callback passed")
    try:
        with (directory / "main_case.fory").open("rb") as stream:
            model = codec.load(stream)
        assert model.callback(2) == 3
    except Exception as exc:
        print("pyfory failure", type(exc).__name__, str(exc))
        raise
    print("pyfory callback passed")


if __name__ == "__main__":
    main()
