# Pyfory reload review scripts

Run these scripts from the repository root with the local virtual environment.
They use a single process; no MPI launcher is needed.

```sh
.venv/bin/python pyfory_review_models.py
.venv/bin/python mpisppy/tests/pyfory/pyfory_review_models.py
.venv/bin/python mpisppy/tests/pyfory/pyfory_time_case.py
```

The first two commands run the same maintained model probe. The time case
compares pyfory with dill and asserts that the full `datetime.time` value is
restored.

The callback case uses two separate Python processes and a shared temporary
directory. The reader asserts that both backends restore a callback defined
in the writer's `__main__` module.

```sh
case_dir=$(mktemp -d)
.venv/bin/python mpisppy/tests/pyfory/pyfory_main_writer.py "$case_dir"
.venv/bin/python mpisppy/tests/pyfory/pyfory_main_reader.py "$case_dir"
```

The power CEP case compares a serial checkpoint and resume with an
uninterrupted run for both backends. It requires the solver configured by
`mpisppy.tests.test_checkpoint`.

```sh
.venv/bin/python mpisppy/tests/pyfory/pyfory_power_cep_resume.py
```
