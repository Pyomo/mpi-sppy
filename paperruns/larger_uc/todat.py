import pandas as pd
import numpy as np

import os
from shutil import copyfile

input_filename = "./wind_actuals_ARMA/simulations_of_target_mape_45.1.csv"

scenario_data = pd.read_csv(input_filename)

scale_factor = 0.05

num_scenarios = 250

columns = scenario_data.iloc[:,1:num_scenarios+1]

OUTDIR = "NewInstance"

os.mkdir(OUTDIR)

copyfile("RootNode.dat", OUTDIR+os.sep+"RootNode.dat")

for scenario_index in range(0,num_scenarios):

    outfile = open(OUTDIR+os.sep+"Node"+str(scenario_index+1)+".dat",'w')
    
    this_column = columns.iloc[:,scenario_index:scenario_index+1].T.to_numpy().reshape((25,))
    print(scenario_index,"THIS COLUMN=",this_column)
    print("AVERAGE WIND=",np.average(this_column))

    print("param MinNondispatchablePower := ", file=outfile)
    
    for hour in range(0,24):
        wind_this_hour = 0.0 # this_column[hour]
        print("WIND", hour+1, wind_this_hour, file=outfile)
    for hour in range(24,48):
        wind_this_hour = 0.0 # this_column[23]
        print("WIND", hour+1, wind_this_hour, file=outfile)        

    print(";", file=outfile)
    print("\n", file=outfile)

    print("param MaxNondispatchablePower := ", file=outfile)
    
    for hour in range(0,24):
        wind_this_hour = this_column[hour] * scale_factor
        print("WIND", hour+1, wind_this_hour, file=outfile)
    for hour in range(24,48):
        wind_this_hour = this_column[23] * scale_factor
        print("WIND", hour+1, wind_this_hour, file=outfile)        

    print(";", file=outfile)
    print("\n", file=outfile)

    outfile.close()

copyfile("ScenarioStructureBase.dat", OUTDIR+os.sep+"ScenarioStructure.dat")

s_outfile = open(OUTDIR+os.sep+"ScenarioStructure.dat",'a')

print("\nset Stages := FirstStage SecondStage ;\n", file=s_outfile)

print("set Nodes := ", file=s_outfile)
print("RootNode", file=s_outfile)
for i in range(1,num_scenarios+1):
    print("Node"+str(i), file=s_outfile)
print(";\n", file=s_outfile)

print("param NodeStage := ", file=s_outfile)
print("RootNode FirstStage", file=s_outfile)
for i in range(1,num_scenarios+1):
    print("Node"+str(i)+" SecondStage", file=s_outfile)
print(";\n", file=s_outfile)

print("set Children[RootNode] := ", file=s_outfile)
for i in range(1,num_scenarios+1):
    print("Node"+str(i), file=s_outfile)
print(";\n", file=s_outfile)

print("param ConditionalProbability :=", file=s_outfile)
print("RootNode 1.0", file=s_outfile)
for i in range(1,num_scenarios+1):
    print("Node"+str(i)+" "+str(1.0/float(num_scenarios)), file=s_outfile)
print(";\n", file=s_outfile)

print("set Scenarios := ", file=s_outfile)
for i in range(1,num_scenarios+1):
    print("Scenario"+str(i), file=s_outfile)
print(";\n", file=s_outfile)

print("param ScenarioLeafNode := ", file=s_outfile)
for i in range(1,num_scenarios+1):
    print("Scenario"+str(i)+" Node"+str(i), file=s_outfile)
print(";\n", file=s_outfile)
    
s_outfile.close()


