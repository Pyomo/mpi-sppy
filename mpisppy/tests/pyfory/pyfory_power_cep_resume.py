import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mpisppy.extensions.checkpointer import Checkpointer
from mpisppy.opt.ph import PH
from mpisppy.tests.test_checkpoint import _options, _primal_snapshot

sys.path.insert(0, str(ROOT / "examples/power_cep"))
import power_cep

data = power_cep._load_system_data()
names = ["scen0", "scen1"]
kwargs = {"system_data": data, "num_scens": 2, "seed": 0}


def make_ph(iterations, *, checkpoint_dir=None, resume_from=None,
            backend="pyfory-reload"):
    options = _options(
        iterations, ckpt_dir=checkpoint_dir, resume_from=resume_from,
        checkpoint_backend=backend,
        checkpoint_structural_cfg={"model": "power_cep", "num_scens": 2,
                                   "seed": 0})
    return PH(options, names, power_cep.scenario_creator,
              power_cep.scenario_denouement,
              scenario_creator_kwargs=kwargs,
              extensions=Checkpointer if checkpoint_dir else None)


with tempfile.TemporaryDirectory() as directory:
    ref = make_ph(2)
    ref.ph_main()
    expected = _primal_snapshot(ref)
    results = {}
    for backend in ("dill-reload", "pyfory-reload"):
        checkpoint_dir = str(pathlib.Path(directory) / backend)
        writer = make_ph(1, checkpoint_dir=checkpoint_dir, backend=backend)
        writer.ph_main()
        resumed = make_ph(1, resume_from=checkpoint_dir, backend=backend)
        resumed.ph_main()
        assert resumed._resumed_from_checkpoint
        actual = _primal_snapshot(resumed)
        results[backend] = actual
        mismatches = [(key, expected[key], actual.get(key))
                      for key in expected if expected[key] != actual.get(key)]
        print(backend, "mismatches vs uninterrupted", len(mismatches),
              mismatches[:10])
    mismatches = [(key, results["dill-reload"].get(key),
                   results["pyfory-reload"].get(key))
                  for key in expected if results["dill-reload"].get(key)
                  != results["pyfory-reload"].get(key)]
    print("pyfory vs dill mismatches", len(mismatches), mismatches[:10])
    assert results["pyfory-reload"] == results["dill-reload"]
