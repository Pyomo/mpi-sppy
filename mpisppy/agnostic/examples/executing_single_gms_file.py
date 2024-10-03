import os
import sys
import gams
import gamspy_base

# This can be useful to execute a single gams file and print its values

this_dir = os.path.dirname(os.path.abspath(__file__))

gamspy_base_dir = gamspy_base.__path__[0]

w = gams.GamsWorkspace(working_directory=this_dir, system_directory=gamspy_base_dir)

model = w.add_job_from_file("transport_ef")

model.run()#output=sys.stdout)

obj_dict = {}
for record in model.out_db.get_variable('z_stoch'):
    obj_dict[tuple(record.keys)] = record.get_level()

x_dict = {}
for record in model.out_db.get_variable('x'):
    x_dict[tuple(record.keys)] = record.get_level()
print(f"{x_dict=}, {obj_dict=}")
print(f"obj_val = {model.out_db.get_variable('z_average').find_record().get_level()}")
