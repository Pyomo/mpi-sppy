###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
''' An extension to track the PH object (mpisppy.opt.ph.PH) during execution.
    Must use the PH object for this to work
'''
import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from mpisppy.extensions.extension import Extension
from mpisppy.cylinders.spoke import ConvergerSpokeType
from mpisppy.cylinders.spoke import Spoke
from mpisppy.cylinders.reduced_costs_spoke import ReducedCostsSpoke 

class TrackedData():
    ''' A class to manage the data for a single variable (e.g. gaps, bounds, etc.)
    '''
    def __init__(self, name, folder, plot=False, verbose=False):
        self.name = name
        self.folder = folder
        self.plot = plot
        self.verbose = verbose
        self.columns = None
        self.df = None
        self.fname = None
        self.plot_fname = None
        self.seen_iters = set()

    def initialize_fnames(self, name=None):
        """ Initialize filenames for saving and plotting
        """
        name = self.name if name is None else name
        name = name[:-4] if name.endswith('.csv') else name

        self.fname = os.path.join(self.folder, f'{name}.csv')
        # LRL: Encountered a bug where plot = False, but plots where still generated,
        # leading to error where plot_fname is None. So, always setting it here.
        #if self.plot:
        self.plot_fname = os.path.join(self.folder, f'{name}.png')

    def initialize_df(self, columns):
        """ Initialize the dataframe for saving the data and write out the column names
        as future rows will be appended to the dataframe
        """
        self.columns = columns
        self.df = pd.DataFrame(columns=columns)
        self.df.to_csv(self.fname, index=False, header=True)

    def add_row(self, row):
        """ Add a row to the dataframe;
        Assumes the first column is the iteration number if row is a list
        """
        assert len(row) == len(self.columns)
        if isinstance(row, dict):
            row_iter = row['iteration']
        elif isinstance(row, list):
            row_iter = row[0]
        else:
            raise RuntimeError("row must be a dict or list")
        if row_iter in self.seen_iters:
            if self.verbose:
                print(f"WARNING: Iteration {row_iter} already seen for {self.name}")
            return
        self.seen_iters.add(row_iter)
        # since append is deprecated
        new_dict = pd.DataFrame([row], columns=self.columns)
        if len(self.df) == 0:
            self.df = new_dict
        else:
            self.df = pd.concat([self.df, new_dict], ignore_index=True)

    def write_out_data(self):
        """ Write out the cached data to csv file and clear the cache
        """
        self.df.to_csv(self.fname, mode='a', header=False, index=False)
        self.df = pd.DataFrame(columns=self.columns)

class PHTracker(Extension):
    """ Class for tracking the PH algorithm

    NOTE:
    Can generalize this code to beyond PH by subclassing TrackedData for each
    variable type, e.g. TrackedGaps, TrackedBounds, etc. and then adding
    the initialize_*, add_* and plot_* functions to the respective classes.
    This seems like a lot of classes for the benefit of one extension, so will hold off.

    Must pass cylinder_name in options if multiple cylinders of the same class are being used
    """
    def __init__(self, opt):
        """
        Args:
            PH object (mpisppy.opt.ph.PH)
        """
        super().__init__(opt)
        self.verbose = self.opt.options["verbose"]
        if 'phtracker_options' in self.opt.options:
            self.tracker_options = self.opt.options["phtracker_options"]
        else:
            raise RuntimeError("phtracker_options not specified in options dict")

        self.results_folder = self.tracker_options.get("results_folder", "results")
        self.save_every = self.tracker_options.get("save_every", 1)
        self.write_every = self.tracker_options.get("write_every", 3)
        self._rank = self.opt.cylinder_rank
        self._reduce_types = ['nonants', 'duals', 'scen_gaps']

        self._track_var_to_func = {
            'gaps': {'track': self.add_gaps,
                     'finalize': self.plot_xbars_bounds_gaps},
            'bounds': {'track': self.add_bounds,
                       'finalize': self.plot_xbars_bounds_gaps},
            'nonants': {'track': self.add_nonants,
                        'finalize': self.plot_nonants_sgaps_duals},
            'duals': {'track': self.add_duals,
                      'finalize': self.plot_nonants_sgaps_duals},
            'xbars': {'track': self.add_xbars,
                      'finalize': self.plot_xbars_bounds_gaps},
            'scen_gaps': {'track': self.add_scen_gaps,
                          'finalize': self.plot_nonants_sgaps_duals},
            'reduced_costs': {'track': self.add_rc,
                              'finalize': self.plot_rc}
        }

        # will initialize these after spcomm is initialized
        self.spcomm = None
        self.cylinder_folder = None
        self.track_dict = None
        self.finished_init = False

    def finish_init(self):
        """ Finish initialization of the extension as we need the spcomm object
        to be initialized and the tracker is initialized with opt before spcomm
        """
        self.spcomm = self.opt.spcomm
        cylinder_name = self.tracker_options.get(
            "cylinder_name", type(self.spcomm).__name__)
        self.cylinder_folder = os.path.join(self.results_folder, cylinder_name)
        track_types = ['gaps', 'bounds', 'nonants', 'duals', 'xbars', 'scen_gaps', 'reduced_costs']

        self.track_dict = {}
        for t in track_types:
            if self.tracker_options.get(f'track_{t}', False):
                val = True
                # only rank 0 needs to create tracker objects; other ranks need to
                # know what data is being tracked
                if self._rank == 0:
                    plot = self.tracker_options.get(f'plot_{t}', False)
                    val = TrackedData(t, self.cylinder_folder, plot, self.verbose)
                    user_fname = self.tracker_options.get(f"{t}_fname", None)
                    val.initialize_fnames(name=user_fname)

                self.track_dict[t] = val

        if self._rank == 0:
            self.verify_tracking()
            os.makedirs(self.cylinder_folder, exist_ok=True)

        for t in self.track_dict.keys():
            self.initialize_df_columns(t)
        self.finished_init = True

    @property
    def curr_iter(self):
        """ Get the current iteration number

        NOTE: This should probably made less ad hoc
        _PHIter could be inaccurate in that most spokes currently make
        specific function calls rather than ph_main(). Therefore, _PHIter
        may not be up to date.
        """
        if hasattr(self.spcomm, 'A_iter'):
            return self.spcomm.A_iter
        if hasattr(self.spcomm, 'dk_iter'):
            return self.spcomm.dk_iter
        if hasattr(self.opt, '_PHIter'):
            return self.opt._PHIter
        raise RuntimeError("Iteration not found")

    def verify_tracking(self):
        """ Verify that the user has specified the correct tracking options given
            the spcomm object
        """
        if 'gaps' in self.track_dict or 'bounds' in self.track_dict:
            if isinstance(self.spcomm, Spoke) and \
                not hasattr(self.spcomm, 'hub_outer_bound') and \
                not hasattr(self.spcomm, 'hub_inner_bound'):

                raise RuntimeError("Cannot access hub gaps without passing"
                                   " them to spcomm")

    def get_var_names(self, xbar=False):
        """ Get the names of the variables
        """
        var_names = []
        for (sname, model) in self.opt.local_scenarios.items():
            for node in model._mpisppy_node_list:
                for var in node.nonant_vardata_list:
                        var_names.append(var.name if xbar else (sname, var.name))
            if xbar:
                break

        return var_names

    def get_scen_colnames(self):
        """ Get the names of the scenarios
        """
        scen_names = [(sname, b) for sname in self.opt.local_scenarios.keys()
                      for b in ['ub', 'lb']]
        return scen_names

    def initialize_df_columns(self, track_var):
        """ Create dataframes for saving the data by defining the columns
        """
        if (track_var not in self._reduce_types) and self._rank != 0:
            return

        if track_var == 'gaps':
            df_columns = ['hub abs. gap', 'hub rel. gap']
            if isinstance(self.spcomm, Spoke):
                df_columns += ['spoke abs. gap', 'spoke rel. gap']
        elif track_var == 'bounds':
            df_columns = ['hub upper bound', 'hub lower bound']
            if isinstance(self.spcomm, Spoke):
                df_columns += ['spoke bound']
        elif track_var == 'nonants':
            df_columns = self.get_var_names()
        elif track_var == 'duals':
            df_columns = self.get_var_names()
        elif track_var == 'xbars':
            df_columns = self.get_var_names(xbar=True)
        elif track_var == 'scen_gaps':
            df_columns = self.get_scen_colnames()
        elif track_var == 'reduced_costs':
            df_columns = self.get_var_names(xbar=True)
        else:
            raise RuntimeError("track_var not recognized")

        if self._rank == 0 and track_var not in self._reduce_types:
            df_columns.insert(0, 'iteration')
            self.track_dict[track_var].initialize_df(df_columns)
        else:
            comm = self.opt.comms['ROOT']
            df_columns = comm.gather(df_columns, root=0)
            if self._rank == 0:
                df_columns = df_columns[0]
                df_columns.insert(0, 'iteration')
                self.track_dict[track_var].initialize_df(df_columns)

    def _add_data_and_write(self, track_var, data, gather=True, final=False):
        """ Gather the data from all ranks and write it out
        Args:
            track_var (str): the variable to track
            data (dict): the data to write out
            gather (bool): whether to gather the data or not
        """
        if gather and track_var not in self._reduce_types:
            if self.verbose:
                print(f"WARNING: Cannot gather {track_var} data; not a reduce type")
            return

        if gather:
            comm = self.opt.comms['ROOT']
            data = comm.gather(data, root=0)

            # LRL: Bugfix to only get gathered data on rank 0.
            if self._rank == 0:
                data = data[0]

        if isinstance(data, dict):
            data['iteration'] = self.curr_iter
        elif isinstance(data, list):
            data.insert(0, self.curr_iter)

        if self._rank == 0:
            self.track_dict[track_var].add_row(data)
            if final or self.curr_iter % self.write_every == 0:
                self.track_dict[track_var].write_out_data()

    def _get_bounds(self):
        spoke_bound = None
        if isinstance(self.spcomm, Spoke):
            hub_inner_bound = self.spcomm.hub_inner_bound
            hub_outer_bound = self.spcomm.hub_outer_bound
            spoke_bound = self.spcomm.bound
        else:
            hub_inner_bound = self.spcomm.BestInnerBound
            hub_outer_bound = self.spcomm.BestOuterBound
        return hub_outer_bound, hub_inner_bound, spoke_bound
    
    def _get_rc(self):
        if not isinstance(self.spcomm, ReducedCostsSpoke):
            return None
        reduced_costs = self.spcomm.rc
        return reduced_costs

    def _ob_ib_process(self, ob, ib):
        """ process the outer and inner bounds
        Args:
            ob (float): outer bound
            ib (float): inner bound
        Returns:
            ub (float): upper bound
            lb (float): lower bound
        """
        if self.opt.is_minimizing:
            ub = ib
            lb = ob
        else:
            ub = ob
            lb = ib
        return ub, lb

    def add_bounds(self, final=False):
        """ add iteration bounds to row
        """
        if self._rank != 0:
            return
        hub_outer_bound, hub_inner_bound, spoke_bound = self._get_bounds()
        upper_bound, lower_bound = self._ob_ib_process(hub_outer_bound, hub_inner_bound)

        if isinstance(self.spcomm, Spoke):
            row = [upper_bound, lower_bound, spoke_bound]
        else:
            row = [upper_bound, lower_bound]
        self._add_data_and_write('bounds', row, gather=False, final=final)

    def _compute_rel_gap(self, abs_gap, outer_bound):
        """ compute the relative gap using outer bound as the denominator
        """
        ## define by the best solution, as is common
        nano = float("nan")  # typing aid
        if (
            abs_gap != nano
            and abs_gap != float("inf")
            and abs_gap != float("-inf")
            and outer_bound != nano
            and outer_bound != 0
        ):
            rel_gap = abs_gap / abs(outer_bound)
        else:
            rel_gap = float("inf")

        return rel_gap

    def add_gaps(self, final=False):
        """ add iteration gaps to row; spoke gap is computed relative to if it is
            an outer bound or inner bound spoke
        """
        if self._rank != 0:
            return

        hub_outer_bound, hub_inner_bound, spoke_bound = self._get_bounds()
        upper_bound, lower_bound = self._ob_ib_process(hub_outer_bound, hub_inner_bound)
        hub_abs_gap = upper_bound - lower_bound
        hub_rel_gap = self._compute_rel_gap(hub_abs_gap, hub_outer_bound)

        if isinstance(self.spcomm, Spoke):
            # compute spoke gap relative to hub bound so that negative gap means
            # the spoke is worse than the hub

            if ConvergerSpokeType.OUTER_BOUND in self.spcomm.converger_spoke_types:
                spoke_abs_gap = spoke_bound - hub_outer_bound  \
                    if self.opt.is_minimizing else hub_outer_bound - spoke_bound
                spoke_rel_gap = self._compute_rel_gap(spoke_abs_gap, hub_outer_bound)
            elif ConvergerSpokeType.INNER_BOUND in self.spcomm.converger_spoke_types:
                spoke_abs_gap = hub_inner_bound - spoke_bound \
                    if self.opt.is_minimizing else spoke_bound - hub_inner_bound
                spoke_rel_gap = self._compute_rel_gap(spoke_abs_gap, hub_inner_bound)
            else:
                raise RuntimeError("Converger spoke type not recognized")
            row = [hub_abs_gap, hub_rel_gap, spoke_abs_gap, spoke_rel_gap]
        else:
            row = [hub_abs_gap, hub_rel_gap]
        self._add_data_and_write('gaps', row, gather=False, final=final)

    def add_scen_gaps(self, final=False):
        """ add iteration scenario gaps to row
        """
        s_gaps = {}
        for (sname, scenario) in self.opt.local_scenarios.items():
            ob, ib = None, None
            if hasattr(scenario._mpisppy_data, 'outer_bound'):
                ob = scenario._mpisppy_data.outer_bound
            if hasattr(scenario._mpisppy_data, 'inner_bound'):
                ib = scenario._mpisppy_data.inner_bound
            ub, lb = self._ob_ib_process(ob, ib)
            s_gaps[(sname, 'ub')] = ub
            s_gaps[(sname, 'lb')] = lb

        self._add_data_and_write('scen_gaps', s_gaps, gather=True, final=final)

    def add_nonants(self, final=False):
        """ add iteration nonants to row
        """
        nonants = {}
        for k, s in self.opt.local_scenarios.items():
            for node in s._mpisppy_node_list:
                for var in node.nonant_vardata_list:
                    nonants[(k, var.name)] = var.value

        self._add_data_and_write('nonants', nonants, gather=True, final=final)

    def add_rc(self, final=False):
        """ add iteration reduced costs to rpw
        """
        if self._rank != 0:
            return

        reduced_costs = self._get_rc()

        if reduced_costs is None:
            return
        rc = {}
        sname = list(self.opt.local_scenarios.keys())[0]
        s = self.opt.local_scenarios[sname]
        rc = {xvar.name: reduced_costs[ci]
                for ci, (ndn_i, xvar) in enumerate(s._mpisppy_data.nonant_indices.items())}

        self._add_data_and_write('reduced_costs', rc, gather=False, final=final)

    def add_duals(self, final=False):
        """ add iteration duals to rpw
        """
        local_duals_data = {(sname, var.name):
                            scenario._mpisppy_model.W[node.name, ix]._value
                    for (sname, scenario) in self.opt.local_scenarios.items()
                    for node in scenario._mpisppy_node_list
                    for (ix, var) in enumerate(node.nonant_vardata_list)}

        self._add_data_and_write('duals', local_duals_data, gather=True, final=final)

    def add_xbars(self, final=False):
        """ add iteration xbars to xbars_df
        """
        if self._rank != 0:
            return
        sname = list(self.opt.local_scenarios.keys())[0]
        scenario = self.opt.local_scenarios[sname]
        xbars = {var.name: scenario._mpisppy_model.xbars[node.name, ix]._value
                    for node in scenario._mpisppy_node_list
                    for (ix, var) in enumerate(node.nonant_vardata_list)}

        self._add_data_and_write('xbars', xbars, gather=False, final=final)

    def plot_gaps(self, var):
        ''' plot the gaps; Assumes gaps are saved in a csv file
        '''


        df = pd.read_csv(self.track_dict[var].fname, sep=',')
        df = df.replace([np.inf, -np.inf], np.nan)


        plt.figure(figsize=(10, 6))
        plt.plot(df['iteration'], df['hub abs. gap'], marker='o', label='Hub Abs. Gap')
        plt.plot(df['iteration'], df['hub rel. gap'], marker='o', label='Hub Rel. Gap')
        if isinstance(self.spcomm, Spoke):
            plt.plot(df['iteration'], df['spoke abs. gap'], marker='o', label='Spoke Abs. Gap')
            plt.plot(df['iteration'], df['spoke rel. gap'], marker='o', label='Spoke Rel. Gap')

        plt.xlabel('Iteration')
        plt.ylabel('Value (log scale)')
        plt.title('Absolute and Relative Gaps Over Iterations')
        plt.legend()
        plt.grid(True, which='major', linestyle='-', linewidth='0.5')
        plt.grid(True, which='minor', linestyle='--', linewidth='0.5')

        plt.savefig(self.track_dict[var].plot_fname)
        plt.close()

    def plot_nonants_sgaps_duals(self, var):
        ''' plot the nonants/scene gaps/dual values; Assumes var is saved in a csv file
        '''
        if self._rank != 0:
            return

        if var not in ['nonants', 'duals', 'scen_gaps']:
            raise RuntimeError("var must be either nonants or duals")

        df = pd.read_csv(self.track_dict[var].fname, sep=',')
        df = df.replace([np.inf, -np.inf], np.nan)
        df.columns = [col if col == 'iteration' else ast.literal_eval(col) for col in df.columns]

        plt.figure(figsize=(16, 6))  # Adjust the figure size as needed
        column_names = df.columns[1:]

        for col in column_names:
            scenario, variable = col

            label = f'{scenario} {variable}'
            plt.plot(df['iteration'], df[col], label=label)

        plt.legend(loc='lower right')
        plt.xlabel('Iteration')
        plt.ylabel(f'{var.capitalize()} Value')
        plt.grid(True)
        plt.savefig(self.track_dict[var].plot_fname)
        plt.close()

    def plot_xbars_bounds_gaps(self, var):
        ''' plot the xbar/bounds/gaps values;
            Assumes xbars/bounds/gaps are saved in a csv file
        '''
        if self._rank != 0:
            return

        if var not in ['xbars', 'bounds', 'gaps']:
            raise RuntimeError('var must be either xbars, bounds, or gaps')

        df = pd.read_csv(self.track_dict[var].fname, sep=',')
        df = df.replace([np.inf, -np.inf], np.nan)

        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        column_names = df.columns[1:]

        for col in column_names:
            plt.plot(df['iteration'], df[col], label=col.capitalize())

        if var == 'gaps':
            df = df.dropna()
            nonzero_gaps = df[df['hub rel. gap'] != 0]['hub rel. gap']
            if len(nonzero_gaps) > 0:
                threshold = np.percentile(nonzero_gaps, 10)
                plt.yscale('symlog', linthresh=threshold)
            else:
                if self.verbose:
                    print("WARNING: No nonzero gaps to compute threshold")

        plt.xlabel('Iteration')
        plt.ylabel(f'{var.capitalize()} values')
        plt.title(f'{var.capitalize()} Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.track_dict[var].plot_fname)
        plt.close()

    def plot_rc(self, var):
        if self._rank != 0:
            return

        if var not in ['reduced_costs']:
            raise RuntimeError('var must be reduced_costs')

        df = pd.read_csv(self.track_dict[var].fname, sep=',')
        df = df.replace([np.inf, -np.inf], np.nan)
        #df.dropna(inplace=True)

        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        column_names = df.columns[1:]

        for col in column_names:
            if not np.isnan(df[col]).all():
                plt.plot(df['iteration'], df[col], label=col.capitalize())

        plt.xlabel('Iteration')
        plt.ylabel(f'{var.capitalize()} values')
        plt.title(f'{var.capitalize()} Over Iterations')
        plt.legend(bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.savefig(self.track_dict[var].plot_fname)
        plt.close()

    def pre_solve_loop(self):
        if not self.finished_init:
            self.finish_init()
        if self.curr_iter % self.save_every == 0:
            for track_var in self.track_dict.keys():
                self._track_var_to_func[track_var]['track']()

    def post_everything(self):
        for track_var in self.track_dict.keys():
                self._track_var_to_func[track_var]['track'](final=True)
                self._track_var_to_func[track_var]['finalize'](var=track_var)
