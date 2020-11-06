# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# January 2020; rho_setter for ccopf from n-1 stuff of a few years ago 

def generator_p_cost(md, g, output_level):
    # g is an egret tuple (num, dict); md is egret model data
    if md.attributes("generator")['p_cost'][g[0]]['cost_curve_type']\
       != 'polynomial':
        print ("rho setter wants polynomial, returning 1")
        return 1
    CCVs = md.attributes("generator")['p_cost'][g[0]]['values']
    a0 = CCVs[2] #???? delete this comment after checking
    a1 = CCVs[1]
    a2 = CCVs[0]
    retval = a0 + a1 * output_level + a2 * output_level**2
    if retval == 0:
        retval = 0.99 
    return retval
        


def ph_rhosetter_callback(scen):
    # pyomo generators: pgen; egret generators: egen
    # we assume all generators are subject to non-anticipativity
    MyRhoFactor = 1.0
    retlist = list()
    numstages = len(scen._PySPnode_list)
    generator_set = scen._egret_md.attributes("generator")
    generator_names = generator_set["names"]
    
    for egen in generator_names:
        for stage in range(1, numstages):
            pgen = scen.stage_models[stage].pg[egen] # pgen is a Var
            
            lb = pgen.lb
            ub = pgen.ub
            mid = lb + (ub - lb)/2.0
            cost_at_mid = generator_p_cost(scen._egret_md, egen, mid)
            rho = cost_at_mid * MyRhoFactor
            print ("dlw debug: pgen={}, rho={}".format(pgen, rho))
            idv = id(pgen)
            retlist.append((idv, rho))
            qgen = scen.stage_models[stage].qg[egen]
            retlist.append((id(qgen), rho))
    return retlist
    """
    ph.setRhoOneScenario(root_node,
                       scenario,
                       symbol_map.getSymbol(scenario_instance.stage1.PG[g]),
                       cost_at_mid * MyRhoFactor)

    ph.setRhoOneScenario(root_node,
                       scenario,
                       symbol_map.getSymbol(scenario_instance.stage1.QG[g]),
                       cost_at_mid * MyRhoFactor)
    """
