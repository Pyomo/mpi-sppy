# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
''' Utilities for reading and writing W and 
    x-bar values in and out of csv files. 

    Written: DLW July 2019
    Modified: DTM Aug 2019

    Could stand to be re-factored a bit.

    When reading/writing the weight files, there are two options.
        
     1. All weights are stored in a single master file. 
        
            Reading: Set "init_W_fname" to the location of the master file
            containing all of the weights.

            Writing: Set "W_fname" to the location of the master file that will
            contain all the weights.

     2. Each scenario's weights are stored in separate, individual files.
        
            Reading: Set "init_W_fname" to the directory containing the weight
            files. This directory must contain one file per scenario, each
            named <scenario_name>_weights.csv. Set "init_separate_W_files" to
            True in the PHoptions dictionary.

            Writing: Set "W_fname" to the directory that will contain the
            weight files (this directory will be created if it does not already
            exist). The weight files will be corretly named in accordance with
            the naming convention above. Set "separate_W_files" to True in the
            PHoptions dictionary.
'''

import pyomo.environ as pyo
import os

''' W utilities '''

def write_W_to_file(PHB, fname, sep_files=False):
    '''
    Args:
        PHB (PHBase object) -- Where the W values live
        fname (str) -- name of file to which we write.
        sep_files (bool, optional) -- If True, one file will be written for
            each scenario, rather than one master file. The names of the files
            are the names of the scenarios.

    Notes:
        All ranks pass their information to rank 0, which then writes a single
        file. This can apparently be accomplished using Collective MPI I/O (see
        https://mpi4py.readthedocs.io/en/stable/tutorial.html#mpi-io), but I am
        lazy and doing this for now.
    '''

    if (sep_files):
        for (sname, scenario) in PHB.local_scenarios.items():
            scenario_Ws = {var.name: pyo.value(scenario._Ws[node.name, ix])
                for node in scenario._PySPnode_list
                for (ix, var) in enumerate(node.nonant_vardata_list)}
            scenario_fname = os.path.join(fname, sname + '_weights.csv')
            with open(scenario_fname, 'w') as f:
                for (vname, val) in scenario_Ws.items():
                    row = ','.join([vname, str(val)]) + '\n'
                    f.write(row)
    else:
        local_Ws = {(sname, var.name): pyo.value(scenario._Ws[node.name, ix])
                    for (sname, scenario) in PHB.local_scenarios.items()
                    for node in scenario._PySPnode_list
                    for (ix, var) in enumerate(node.nonant_vardata_list)}
        comm = PHB.comms['ROOT']
        Ws = comm.gather(local_Ws, root=0)
        if (PHB.rank == 0):
            with open(fname, 'a') as f:
                for W in Ws:
                    for (key, val) in W.items():
                        sname, vname = key[0], key[1]
                        row = ','.join([sname, vname, str(val)]) + '\n'
                        f.write(row)

def set_W_from_file(fname, PHB, rank, sep_files=False):
    ''' 
    Args:
        fname (str) -- if sep_files=False, file containing the dual weights.
            Otherwise, path of the directory containing the dual weight files
            (one per scenario).
        PHB (PHBase object) -- Where the W values will be put
        rank (int) -- rank number
        sep_files (bool, optional) -- If True, attempt to read weights from
            individual files, one per scenario. The files must be contained in
            the same directory, and must be named <sname>_weights.csv for each
            scenario name <sname>.
    
    Notes:
        Calls _check_W, which ensures that all required values were specified,
        and that the specified weights satisfy the dual feasibility condition 
        sum_{s\in S} p_s * w_s = 0.
    '''
    scenario_names_local  = list(PHB.local_scenarios.keys())
    scenario_names_global = PHB.all_scenario_names

    if (sep_files):
        w_val_dict = dict()
        for sname in scenario_names_local:
            scenario_fname = os.path.join(fname, sname + '_weights.csv')
            w_val_dict[sname] = _parse_W_csv_single(scenario_fname)
    else:
        w_val_dict = _parse_W_csv(fname, scenario_names_local,
                                    scenario_names_global, rank)

    _check_W(w_val_dict, PHB, rank)

    mp = {(sname, var.name): (node.name, ix)
            for (sname, scenario) in PHB.local_scenarios.items()
            for node in scenario._PySPnode_list
            for (ix,var) in enumerate(node.nonant_vardata_list)}

    for (sname, d) in w_val_dict.items():
        for vname in d.keys():
            scenario = PHB.local_scenarios[sname]
            node_name, ix = mp[sname, vname]
            scenario._Ws[node_name, ix] = w_val_dict[sname][vname]

def _parse_W_csv_single(fname):
    ''' Read a file containing the weights for a single scenario. The file must
        be formatted as 

        variable_name,variable_value

        (comma separated). Lines beginning with a "#" are treated as comments
        and ignored.
    '''
    if (not os.path.exists(fname)):
        raise RuntimeError('Could not find file {fn}'.format(fn=fname))
    results = dict()
    with open(fname, 'r') as f:
        for line in f:
            if (line.startswith('#')):
                continue
            line  = line.split(',')
            vname = ','.join(line[:-1])
            wval  = float(line[-1])
            results[vname] = wval
    return results

def _parse_W_csv(fname, scenario_names_local, scenario_names_global, rank):
    ''' Read a csv file containing weight information. 
        
        Args:
            fname (str) -- Filename of csv file to read
            scenario_names_local (list of str) -- List of local scenario names
            scenario_names_global (list of str) -- List of global scenario
                names (i.e. all the scenario names in the entire model across
                all ranks--each PHBase object stores this information).

        Return:
            results (dict) -- Doubly-nested dict mapping 
                results[scenario_name][var_name] --> weight value (float)
    
        Notes:
            This function is only called if sep_files=False, i.e., if all of
            the weights are stored in a single master file. The file must be
            formatted as:

            scenario_name,variable_name,weight_value

            Rows that begin with a "#" character are treated as comments.
            The variable names _may_ contain commas (confusing, but simpler for
            the user)

            Raises a RuntimeError if there are any missing scenarios. Prints a
            warning if there are any extra scenarios.

            When this function returns, we are certain that 
                results.keys() == PHB.local_scenarios.keys()

            When run in parallel, this method requires multiple ranks to open
            and read from the same file simultaneously. Apparently there are
            safer ways to do this using MPI collective communication, but since
            all we're doing here is reading files, I'm being lazy and doing it
            this way.
    '''
    results = dict()
    seen = {name: False for name in scenario_names_local}
    with open(fname, 'r') as f:
        for line in f:
            if (line.startswith('#')):
                continue
            line  = line.split(',')
            sname = line[0]
            vname = ','.join(line[1:-1])
            wval  = float(line[-1])
            
            if (sname not in scenario_names_global):
                if (rank == 0):
                    print('WARNING: Ignoring unknown scenario name', sname)
                continue
            if (sname not in scenario_names_local):
                continue
            if (sname in results):
                results[sname][vname] = wval
            else:
                seen[sname] = True
                results[sname] = {vname: wval}
    missing = [name for (name,is_seen) in seen.items() if not is_seen]
    if (missing):
        raise RuntimeError('rank ' + str(rank) +' could not find the following '
                'scenarios in the provided weight file: ' + ', '.join(missing)) 
        
    return results
            
def _check_W(w_val_dict, PHB, rank):
    '''
    Args:
        w_val_dict (dict) -- doubly-nested dict mapping 
            w_val_dict[scenario_name][variable_name] = weight value.
        PHB (PHBase object) -- PHBase object
        rank (int) -- local rank

    Notes:
        Checks for three conditions:
         
         1. Missing variables --> raises a RuntimeError
         2. Extra variables --> prints a warning
         3. Dual feasibility --> raises a RuntimeError
    '''
    # By this point, we are certain that
    # w_val_dict.keys() == PHB.local_scenarios.keys()
    for (sname, scenario) in PHB.local_scenarios.items():
        vn_model = set([var.name for node in scenario._PySPnode_list
                                 for var  in node.nonant_vardata_list])
        vn_provided = set(w_val_dict[sname].keys())
        diff = vn_model.difference(vn_provided)
        if (diff):
            raise RuntimeError(sname + ' is missing '
                'the following variables: ' + ', '.join(list(diff)))
        diff = vn_provided.difference(vn_model)
        if (diff):
            print('Removing unknown variables:', ', '.join(list(diff)))
            for vname in diff:
                w_val_dict[sname].pop(vname, None)
        
    # At this point, we are sure that every local 
    # scenario has the same set of variables
    probs = {name: model.PySP_prob for (name, model) in
                    PHB.local_scenarios.items()}

    checks = dict()
    for vname in vn_model: # Ensured vn_model = vn_provided
        checks[vname] = sum(probs[name] * w_val_dict[name][vname]
                            for name in PHB.local_scenarios.keys())

    checks = PHB.comms['ROOT'].gather(checks, root=0)
    if (rank == 0):
        for vname in vn_model:
            dual = sum(c[vname] for c in checks)
            if (abs(dual) > 1e-7):
                raise RuntimeError('Provided weights do not satisfy '
                    'dual feasibility: \sum_{scenarios} prob(s) * w(s) != 0. '
                    'Error on variable ' + vname)

''' X-bar utilities '''

def write_xbar_to_file(PHB, fname):
    '''
    Args:
        PHB (PHBase object) -- Where the W values live
        fname (str) -- name of file to which we write.

    Notes:
        Each scenario maintains its own copy of xbars. We only need to write
        one of them to the file (i.e. no parallelism required).
    '''
    if (PHB.rank != 0):
        return
    sname = list(PHB.local_scenarios.keys())[0]
    scenario = PHB.local_scenarios[sname]
    xbars = {var.name: pyo.value(scenario._xbars[node.name, ix])
                for node in scenario._PySPnode_list
                for (ix, var) in enumerate(node.nonant_vardata_list)}
    with open(fname, 'a') as f:
        for (var_name, val) in xbars.items():
            row = ','.join([var_name, str(val)]) + '\n'
            f.write(row)

def set_xbar_from_file(fname, PHB):
    ''' Read all of the csv files in a directory and use them to populate
        _xbars and _xsqbars

    Args:
        fname (str) -- file containing the dual weights
        PHB (PHBase object) -- Where the W values will be put

    Notes:
        Raises a RuntimeError if the provided file is missing any values for
        xbar (i.e. does not assume a default value for missing variables).
    '''
    xbar_val_dict = _parse_xbar_csv(fname)

    if (PHB.rank == 0):
        _check_xbar(xbar_val_dict, PHB)

    for (sname, scenario) in PHB.local_scenarios.items():
        for node in scenario._PySPnode_list:
            for (ix,var) in enumerate(node.nonant_vardata_list):
                val = xbar_val_dict[var.name]
                scenario._xbars[node.name, ix] = val
                scenario._xsqbars[node.name, ix] = val * val

def _parse_xbar_csv(fname):
    ''' Read a csv file containing weight information. 
        
        Args:
            fname (str) -- Filename of csv file to read
        Return:
            results (dict) -- Dict mapping var_name --> variable value (float)
    
        Notes:
            The file must be formatted as:

            variable_name,value

            Rows that begin with a "#" character are treated as comments.
            The variable names _may_ contain commas (confusing, but simpler for
            the user)

            When run in parallel, this method requires multiple ranks to open
            and read from the same file simultaneously. Apparently there are
            safer ways to do this using MPI collective communication, but since
            all we're doing here is reading files, I'm being lazy and doing it
            this way.
    '''
    results = dict()
    with open(fname, 'r') as f:
        for line in f:
            if (line.startswith('#')):
                continue
            line  = line.split(',')
            vname = ','.join(line[:-1])
            val  = float(line[-1])
            
            results[vname] = val

    return results

def _check_xbar(xbar_val_dict, PHB):
    ''' Make sure that a value was provided for every non-anticipative
        variable. If any extra variable values were provided in the input file,
        this function prints a warning.
    '''
    sname = list(PHB.local_scenarios.keys())[0]
    scenario = PHB.local_scenarios[sname]
    var_names = set([var.name for node in scenario._PySPnode_list
                          for var  in node.nonant_vardata_list])
    provided_vars = set(xbar_val_dict.keys())
    set1 = var_names.difference(provided_vars)
    if (set1):
        raise RuntimeError('Could not find the following required variable '
            'values in the provided input file: ' + ', '.join([v for v in set1]))
    set2 = provided_vars.difference(var_names)
    if (set2):
        print('Ignoring the following variables values provided in the '
              'input file: ' + ', '.join([v for v in set2]))
