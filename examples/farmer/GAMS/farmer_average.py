import os
import sys
import gams
import gamspy.utils as utils

this_dir = os.path.dirname(os.path.abspath(__file__))

w = gams.GamsWorkspace(working_directory=this_dir, system_directory=utils._getGAMSPyBaseDirectory())

model = w.add_job_from_file("farmer_average.gms")

model.run(output=sys.stdout)
