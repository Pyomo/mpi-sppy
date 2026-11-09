"""Write a model with a callback defined in this script's __main__ module."""

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import dill
import pyomo.environ as pyo
from mpisppy.utils import pyfory_model_codec as codec


def callback(value):
    return value + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    directory = parser.parse_args().directory
    directory.mkdir(parents=True, exist_ok=True)

    model = pyo.ConcreteModel()
    model.callback = callback
    fory_path = directory / "main_case.fory"
    dill_path = directory / "main_case.dill"
    with fory_path.open("wb") as stream:
        codec.dump(model, stream)
    with dill_path.open("wb") as stream:
        dill.dump(model, stream)
    with fory_path.open("rb") as stream:
        assert codec.load(stream).callback(2) == 3
    print("writer round trip passed", fory_path, dill_path)


if __name__ == "__main__":
    main()
