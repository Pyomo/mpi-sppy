# This software is distributed under the 3-clause BSD License.
# Utilities to support formation and use of "proper" bundles

import dill  

def dill_pickle(model, fname):
    """ serialize model using dill to file name"""
    global_toc(f"about to pickle to {fname}")
    with open(fname, "wb") as f:
        dill.dump(model, f)
    global_toc(f"done with pickle {fname}")


def dill_unpickle(fname):
    """ load a model from fname"""
    
    global_toc(f"about to unpickle {fname}")
    with open(fname, "rb") as f:
        m = dill.load(f)
    global_toc(f"done with unpickle {fname}")
    return m


def pickle_bundle_parser(parser):
    """ Add command line options for creation and use of "proper" bundles"""
    parser.add_argument('--pickle-bundles-dir',
                        help="Write bundles to a dill pickle files in this dir (default None)",
                        dest='pickle_bundles_dir',
                        type=str,
                        default=None)
    
    parser.add_argument('--unpickle-bundles-dir',
                        help="Read bundles from a dill pickle files in this dir; (default None)",
                        dest='unpickle_bundles_dir',
                        type=str,
                        default=None)
    return parser

def check_args(args):
    """ make sure the pickle bundle args make sense"""    
    assert(args.pickle_bundles_dir is None or args.unpickle_bundles_dir is None)
    if args.pickle_bundles_dir is not None and not os.path.isdir(args.pickle_bundles_dir):
        raise RuntimeError(f"Directory to pickle into not found: {args.pickle_bundles_dir}")
    if args.unpickle_bundles_dir is not None and not os.path.isdir(args.unpickle_bundles_dir):
        raise RuntimeError(f"Directory to load pickle files from not found: {args.unpickle_bundles_dir}")

    
