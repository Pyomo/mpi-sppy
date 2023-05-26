# Utility to allow for a heirarchy of solvers and options
# and perhaps utilities to support special options like threads and mipgap

import mpisppy.utils.sputils as sputils

"""
There is a naming convention that has a root (e.g., PH or Lagranger)
that can be an empty string.

when non-empty:
we look for
root_solver_name  in the Config object and if we find it, we also return sroot_solver_options (or None)
when empty:
we look for
solver_name  in the Config object and if we find it, we also return solver_options (or None)

The options are given as a single string:
    space seperated options with = for arguments 
    (see sputils.option_string_to_dict)

The possible roots are given as an ordered list and we quit when we find one on the Config object.
If the empty root is desired, it must be given in the list.

It is expected that root_args functions on the Config object will support creation of the indexes;
when those are used, thing should go smoothly.

TBD: xxxx should mipgap and solver_threads be special?
   For now, we will let callers deal with them

Note: we are no longer going to allow a dict as the options value. It will have to be a string.

"""

def solver_specification(cfg, prefix="", name_required=True):
    """ Look through cfg to find the soler_name and solver_options.

    Args:
        cfg (Config): options, typically from the command line
        prefix (str or list of str): the prefix strings (e.g. "PH", "", "Lagranger")
        name_required (boolean): throw an error if we don't get a solver name

    Returns:
        sroot (str): the root string found (or None)
        solver_name (str): the solver name (or None)
        solver_options (dict): the options dictionary created from the string
    """
    
    if isinstance(prefix, (list,tuple)):
        root_list = prefix
    else:
        root_list = [prefix, ]

    idx_list = list()
    for sroot in root_list:
        name_idx = "solver_name" if sroot == "" else f"{sroot}_solver_name"
        idx_list.append(name_idx)
        if cfg.get(name_idx) is not None:
            solver_name = cfg[name_idx]
            options_idx = "solver_options" if sroot == "" else f"{sroot}_solver_options"
            ostr = cfg.get(options_idx)
            solver_options = sputils.option_string_to_dict(ostr)  # will return None for None
            break
    else:
        if name_required:
            # Leaving in underscores even though it might confuse command line users
            print(f"\nsolver name arguments checked in Config object = {idx_list}\n")
            raise RuntimeError(f"The Config object did not specify a solver")
    return sroot, solver_name, solver_options
