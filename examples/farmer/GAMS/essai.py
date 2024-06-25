import gams
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum, Sense
import os
import sys
import gamspy_base
import gams.transfer as gt
import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
gamspy_base_dir = gamspy_base.__path__[0]


sys_dir = sys.argv[1] if len(sys.argv) > 1 else None
#ws = gams.GamsWorkspace(working_directory=this_dir, system_directory=gamspy_base_dir)
"""
plants = ["Seattle", "San-Diego"]
markets = ["New-York", "Chicago", "Topeka"]
capacity = {"Seattle": 350.0, "San-Diego": 600.0}
demand = {"New-York": 325.0, "Chicago": 300.0, "Topeka": 275.0}
distance = {
    ("Seattle", "New-York"): 2.5,
    ("Seattle", "Chicago"): 1.7,
    ("Seattle", "Topeka"): 1.8,
    ("San-Diego", "New-York"): 2.5,
    ("San-Diego", "Chicago"): 1.8,
    ("San-Diego", "Topeka"): 1.4,
}

# create new GamsDatabase instance
db = ws.add_database()

# add 1-dimensional set 'i' with explanatory text 'canning plants' to the GamsDatabase
i = db.add_set("i", 1, "canning plants")
for p in plants:
    i.add_record(p)

# add parameter 'a' with domain 'i'
a = db.add_parameter_dc("a", [i], "capacity of plant i in cases")
for p in plants:
    a.add_record(p).value = capacity[p]

# export the GamsDatabase to a GDX file with name 'data.gdx' located in the 'working_directory' of the GamsWorkspace
print("done")"""


container = gt.Container(system_directory=gamspy_base_dir)
ws = gams.GamsWorkspace(working_directory=this_dir, system_directory=gamspy_base_dir)
#job = ws.add_job_from_file("farmer_linear_augmented.gms")
job = ws.add_job_from_file("farmer_average.gms")
job.run()

"""cp = ws.add_checkpoint()
mi = cp.add_modelinstance()

job.run(checkpoint=cp)"""

db = job.out_db

# Transfer sets
for symbol in db:
    domain = [d.name if isinstance(d, gams.GamsSet) else d for d in symbol.domains]
    print(f"{domain=}")
    if isinstance(symbol, gams.GamsSet):
        records = [rec.keys for rec in symbol]
        container.addSet(name=symbol.name, domain=domain, records=records, description=symbol.text)
        print(f"{container[symbol.name]=}")
    
    if isinstance(symbol, gams.GamsParameter):
        print(f"{symbol.name=}")
        print(f"{symbol=}")
        if symbol.number_records == 1:
            for rec in symbol:
                records = rec.value
        else:
            def _key(k):
                #print(f"{k=}")
                if isinstance(k, list):
                    assert len(k)==1, "should only contain one element"
                    return k[0]
                else:
                    return k
            records = [[_key(rec.keys), rec.value] for rec in symbol]
        #records = np.array([(rec.keys, rec.value) for rec in symbol])
        print(f"{records=}")
        container.addParameter(symbol.name, domain=domain, records=records, description=symbol.text)
        

    if isinstance(symbol, gams.GamsVariable):
        print(f"{symbol.name=}")
        if symbol.name == 'profit':
            profit = container.addVariable(symbol.name, domain=domain, description=symbol.text)
        else:
            container.addVariable(symbol.name, domain=domain, description=symbol.text)
    
    if isinstance(symbol, gams.GamsEquation):
        records = [(rec.keys, rec.level) for rec in symbol]
        print(f"{records=}")
        container.addEquation(symbol.name, domain=domain, definition=definition,records=records, description=symbol.text)

b1 = Model(
    container=container,
    name="test1",
    equations=[],
    problem="LP",
    sense=Sense.MIN,
    objective=profit,
)