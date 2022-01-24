# parsers for confidence intervals
# Note to new users: these parsers are for your convenience. It is up to you
# to use the values (i.e. put them in dictionaries and kwargs as appropriate).

def confidence_parser(inparser):
    parser = inparser

    parser.add_argument("--confidence-level",
                        help="1-alpha (default 0.95)",
                        dest="confidence_level",
                        type=float,
                        default=0.95)
    return parser


def MMW_parser(inparser):
    inparser.add_argument('xhatpath',
                          help="path to .npy file with feasible nonant solution xhat")
    
    inparser.add_argument("--MMW-num-batches",
                            help="number of batches used for MMW confidence interval (default 1)",
                            dest="num_batches",
                            type=int,
                            default=1)
    
    inparser.add_argument("--MMW-batch-size",
                            help="batch size used for MMW confidence interval (default None)",
                            dest="batch_size",
                            type=int,
                            default=None) #None means take batch_size=num_scens
    return inparser
    

def sequential_parser(inparser):
    parser = inparser

    parser.add_argument("--sample-size-ratio",
                        help="xhat/gap sample size ratio (default 1)",
                        dest="sample_size_ratio",
                        type=float,
                        default=1.0)

    parser.add_argument("--ArRP",
                        help="How many to pool to comute G and s (default 1)",
                        dest="ArRP",
                        type=int,
                        default=1)

    parser.add_argument("--kf-GS",
                        help="Resampling frequence for CI estimators (default 1)",
                        dest="kf_GS",
                        type=int,
                        default=1)

    parser.add_argument("--kf-xhat",
                        help="Resampling frequence for xhat (default 1)",
                        dest="kf_xhat",
                        type=int,
                        default=1)
    return parser


def BM_parser(inparser):
    # Bayraksan and Morton sequential (relative width)

    parser = inparser

    parser.add_argument("--BM-h",
                        help="Controls width of confidence interval (default 1.75)",
                        dest="BM_h",
                        type=float,
                        default=1.75)

    parser.add_argument("--BM-hprime",
                        help="Controls tradeoff between width and sample size as well as termination (default 0.5)",
                        dest="BM_hprime",
                        type=float,
                        default=0.5)
    
    parser.add_argument("--BM-eps",
                        help="Related to eps-prime, which controls termination (default 0.2)",
                        dest="BM_eps",
                        type=float,
                        default=0.2)
    
    parser.add_argument("--BM-eps-prime",
                        help="Controls termination (default 0.1)",
                        dest="BM_eps_prime",
                        type=float,
                        default=0.1)
    
    parser.add_argument("--BM-p",
                        help="Controls sample size (default 0.1)",
                        dest="BM_p",
                        type=float,
                        default=0.1)

    parser.add_argument("--BM-q",
                        help="Related to sample size growth (default 1.2)",
                        dest="BM_q",
                        type=float,
                        default=1.2)
    return parser


def BPL_parser(inparser):
    # Bayraksan and Pierre-Louis

    parser = inparser

    parser.add_argument("--BPL-eps",
                        help="Controls termination (default 1)",
                        dest="BPL_eps",
                        type=float,
                        default=1)

    parser.add_argument("--BPL-c0",
                        help="Starting sample size (default 20)",
                        dest="BPL_c0",
                        type=int,
                        default=20)

    parser.add_argument("--BPL-n0min",
                        help="Non-zero implies stochastic sampling (default 0)",
                        dest="BPL_n0min",
                        type=int,
                        default=0)
    return parser
