import os
import sys
import gams
import gamspy_base
import examples.farmer.farmer_gams_gen_agnostic as farmer_gams_gen_agnostic

this_dir = os.path.dirname(os.path.abspath(__file__))

gamspy_base_dir = gamspy_base.__path__[0]

ws = gams.GamsWorkspace(working_directory=this_dir, system_directory=gamspy_base_dir)

original_file = "farmer_average.gms"
nonants = "crop"
farmer_gams_gen_agnostic.create_ph_model(original_file, nonants)

model = ws.add_job_from_file("farmer_average_ph")
#model = ws.add_job_from_file("farmer_average_completed")
#model = ws.add_job_from_file("farmer_linear_augmented")

model.run(output=sys.stdout)
