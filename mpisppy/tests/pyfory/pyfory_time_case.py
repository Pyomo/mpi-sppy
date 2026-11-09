"""Compare dill and pyfory reload of a datetime.time model attribute."""

import datetime
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import dill
import pyomo.environ as pyo
from mpisppy.utils import pyfory_model_codec as codec


model = pyo.ConcreteModel()
model.deadline = datetime.time(12, 30, 45, 123456, fold=1)
stream = io.BytesIO()
codec.dump(model, stream)
restored = codec.load(io.BytesIO(stream.getvalue()))
dill_restored = dill.loads(dill.dumps(model))
print("original", model.deadline, model.deadline.fold)
print("pyfory", restored.deadline, restored.deadline.fold)
print("dill", dill_restored.deadline, dill_restored.deadline.fold)
assert dill_restored.deadline == model.deadline
assert dill_restored.deadline.fold == model.deadline.fold
assert restored.deadline == model.deadline
assert restored.deadline.fold == model.deadline.fold
