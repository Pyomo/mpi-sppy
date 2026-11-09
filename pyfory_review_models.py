"""Serial full-model codec smoke checks for representative repo examples.

Run with ``.venv/bin/python pyfory_review_models.py`` from the repository root.
This uses no MPI launcher or solver and writes no checkpoint files.
"""

import io
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import pyomo.environ as pyo

from mpisppy.utils import pyfory_model_codec as codec

ROOT = Path(__file__).resolve().parent
for relative in reversed(("examples/netdes", "examples/hydro",
                          "mpisppy/tests/examples", "examples/power_cep")):
    sys.path.insert(0, str(ROOT / relative))


def snapshot(model):
    return {
        "variables": {
            var.name: (var.value, var.fixed, var.lb, var.ub,
                       var.stale, str(var.domain))
            for var in model.component_data_objects(pyo.Var)
        },
        "parameters": {
            f"{component.name}[{key!r}]":
                pyo.value(value, exception=False)
            for component in model.component_objects(pyo.Param)
            for key, value in component.items()
        },
        "constraints": {
            con.name: (str(con.expr), con.active)
            for con in model.component_data_objects(pyo.Constraint)
        },
        "objectives": {
            obj.name: (str(obj.expr), obj.active, str(obj.sense))
            for obj in model.component_data_objects(pyo.Objective)
        },
        "nodes": [
            (node.name, [var.name for var in node.nonant_vardata_list])
            for node in model._mpisppy_node_list
        ],
        "probability": getattr(model, "_mpisppy_probability", None),
    }


def check(label, model):
    stream = io.BytesIO()
    codec.dump(model, stream)
    restored = codec.load(io.BytesIO(stream.getvalue()))
    before, after = snapshot(model), snapshot(restored)
    assert before == after, f"{label} model state changed during round trip"
    for node in restored._mpisppy_node_list:
        assert all(var.model() is restored for var in node.nonant_vardata_list)
    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "model.fory"
        expected_path = Path(directory) / "expected.pkl"
        model_path.write_bytes(stream.getvalue())
        with expected_path.open("wb") as output:
            pickle.dump(before, output)
        result = subprocess.run(
            [sys.executable, __file__, "--read", str(model_path),
             str(expected_path)], capture_output=True, text=True,
            timeout=120, check=False)
        assert result.returncode == 0, (
            f"{label} fresh-process load failed:\n{result.stdout}{result.stderr}")
    print(label, len(stream.getvalue()), len(after["variables"]), flush=True)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--read":
        with open(sys.argv[2], "rb") as stream:
            restored = codec.load(stream)
        with open(sys.argv[3], "rb") as stream:
            expected = pickle.load(stream)
        assert snapshot(restored) == expected
        for node in restored._mpisppy_node_list:
            assert all(var.model() is restored
                       for var in node.nonant_vardata_list)
        return

    import netdes
    check("netdes", netdes.scenario_creator(
        "scen0", path=str(ROOT / "examples/netdes/data/network-10-10-H-01.dat")))

    import hydro
    check("hydro", hydro.scenario_creator("Scen1", branching_factors=[3, 3]))

    import aircond
    check("aircond", aircond.scenario_creator(
        "scen0", branching_factors=[2, 2], start_seed=1134,
        mu_dev=0.0, sigma_dev=40.0))

    import power_cep
    check("power_cep", power_cep.scenario_creator(
        "scen0", system_data=power_cep._load_system_data(), num_scens=2,
        seed=0))


if __name__ == "__main__":
    main()
