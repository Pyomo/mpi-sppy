###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# to assist with debugging (a lot of stuff here is really needed)
import os
import sys
import gams
import gamspy_base

def create_model(scennum = 2):
    # create a model and return a dictionary describing it

    this_dir = os.path.dirname(os.path.abspath(__file__))

    gamspy_base_dir = gamspy_base.__path__[0]

    ws = gams.GamsWorkspace(working_directory=this_dir, system_directory=gamspy_base_dir)

    job = ws.add_job_from_file("farmer_linear_augmented.gms")

    job.run(output=sys.stdout)

    cp = ws.add_checkpoint()
    mi = cp.add_modelinstance()

    job.run(checkpoint=cp)

    crop = mi.sync_db.add_set("crop", 1, "crop type")

    y = mi.sync_db.add_parameter_dc("yield", [crop,], "tons per acre")

    ph_W = mi.sync_db.add_parameter_dc("ph_W", [crop,], "ph weight")
    xbar = mi.sync_db.add_parameter_dc("xbar", [crop,], "ph average")
    rho = mi.sync_db.add_parameter_dc("rho", [crop,], "ph rho")

    W_on = mi.sync_db.add_parameter("W_on", 0, "activate w term")
    prox_on = mi.sync_db.add_parameter("prox_on", 0, "activate prox term")

    mi.instantiate("simple min negprofit using lp",
        [
            gams.GamsModifier(y),
            gams.GamsModifier(ph_W),
            gams.GamsModifier(xbar),
            gams.GamsModifier(rho),
            gams.GamsModifier(W_on),
            gams.GamsModifier(prox_on),
        ],
    )

    # initialize W, rho, xbar, W_on, prox_on
    crops = [ "wheat", "corn", "sugarbeets" ]
    for c in crops:
        ph_W.add_record(c).value = 0
        xbar.add_record(c).value = 0
        rho.add_record(c).value = 0
    W_on.add_record().value = 0
    prox_on.add_record().value = 0

    # scenario specific data applied
    assert scennum < 3, "three scenarios hardwired for now"
    if scennum == 0:  # below
        y.add_record("wheat").value = 2.0
        y.add_record("corn").value = 2.4
        y.add_record("sugarbeets").value = 16.0
    elif scennum == 1: # average
        y.add_record("wheat").value = 2.5
        y.add_record("corn").value = 3.0
        y.add_record("sugarbeets").value = 20.0
    elif scennum == 2: # above
        y.add_record("wheat").value = 3.0
        y.add_record("corn").value = 3.6
        y.add_record("sugarbeets").value = 24.0

    mi.solve()
    areaVarDatas = list(mi.sync_db["x"])

    print(f"after simple solve for scenario {scennum}")
    for i,v in enumerate(areaVarDatas):
        print(f"{i =}, {v.get_level() =}")

    # In general, be sure to process variables in the same order has the guest does (so indexes match)
    nonant_names_dict = {("ROOT",i): ("x", v.key(0)) for i, v in enumerate(areaVarDatas)}
    gd = {
        "scenario": mi,
        "nonants": {("ROOT",i): v for i,v in enumerate(areaVarDatas)},
        "nonant_fixedness": {("ROOT",i): v.get_lower() == v.get_upper() for i,v in enumerate(areaVarDatas)},
        "nonant_start": {("ROOT",i): v.get_level() for i,v in enumerate(areaVarDatas)},
        "nonant_names": nonant_names_dict,
        "nameset": {nt[0] for nt in nonant_names_dict.values()},
        "probability": "uniform",
        "sense": None,
        "BFs": None,
        "ph" : {
            "ph_W" : {("ROOT",i): p for i,p in enumerate(ph_W)},
            "xbar" : {("ROOT",i): p for i,p in enumerate(xbar)},
            "rho" : {("ROOT",i): p for i,p in enumerate(rho)},
            "W_on" : W_on.first_record(),
            "prox_on" : prox_on.first_record(),
            "obj" : mi.sync_db["negprofit"].find_record(),
            "nonant_lbs" : {("ROOT",i): v.get_lower() for i,v in enumerate(areaVarDatas)},
            "nonant_ubs" : {("ROOT",i): v.get_upper() for i,v in enumerate(areaVarDatas)},
        },
    }

    return gd
        

if __name__ == "__main__":

    scennum = 2
    gd = create_model(scennum=scennum)
    mi = gd["scenario"]
    mi.solve()
    print(f"iter 0 {mi.model_status =}")
    print(f"after solve in main for scenario {scennum}")
    for ndn_i,gxvar in gd["nonants"].items():
        print(f"{ndn_i =}, {gxvar.get_level() =}")
    print("That was bad, but if we do sync_db in the right way, it will be cool")
    ###mi.sync_db["x"]  # not good enough, we need to make the list for some reason
    # the set of names will be just x for farmer 
    print(f"{gd['nameset'] =}")
    for n in gd["nameset"]:    
        list(mi.sync_db[n])
    # I don't really understand this sync_db thing; the iterator seems to have a side-effect
    # But maybe the objects are the objects, so the sync has been done... yes!
    for ndn_i,gxvar in gd["nonants"].items():
        print(f"{ndn_i =}, {gxvar.get_level() =}")
    print("was that good?")
    print("now solve again, get and display levels")
    mi.solve()
    print(f" iter 0 repeat solve {mi.model_status =}")
    for n in gd["nameset"]:    
        list(mi.sync_db[n])
    for ndn_i,gxvar in gd["nonants"].items():
        print(f"{ndn_i =}, {gxvar.get_level() =}")
    print(f" after repeat iter 0 solve {mi.model_status =}")
    
    print("Here is where the trouble starts")
    print("\n Now let's try to simulate an iter 1 solve")
    print(f'{gd["ph"]["prox_on"] =}')
    print(f'{mi.sync_db["prox_on"].find_record() =}')
    #gd["ph"]["prox_on"].set_value(1)
    gd["ph"]["prox_on"].value = 1
    for ndn_i in gd["nonants"]:
        print(f'{gd["ph"]["rho"][ndn_i] =}')
        #gd["ph"]["rho"][ndn_i].set_value(1)
        #gd["ph"]["rho"][ndn_i].value = 1
        ###gd["ph"]["xbar"][ndn_i].set_value(100)
    mi.solve()
    print(f"  regular iter {mi.model_status =}")
    print("Note that the levels do not update with status of 19")
    """
    for n in gd["nameset"]:    
        list(mi.sync_db[n])
    for ndn_i,gxvar in gd["nonants"].items():
        print(f"{ndn_i =}, {gxvar.get_level() =}")
    """
    
    
