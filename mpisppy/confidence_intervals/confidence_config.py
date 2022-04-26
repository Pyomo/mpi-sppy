# configuration data (and parsers) for confidence intervals

from mpisppy.utils.config import add_to_config

def confidence_config():

    add_to_config("confidence_level",
                  description="1-alpha (default 0.95)",
                  domain=float,
                  default=0.95)


def sequential_config():

    add_to_config("sample_size_ratio",
                  description="xhat/gap sample size ratio (default 1)",
                  domain=float,
                  default=1.0)

    add_to_config("ArRP",
                  description="How many to pool to comute G and s (default 1)",
                  domain=int,
                  default=1)

    add_to_config("kf_GS",
                  description="Resampling frequence for CI estimators (default 1)",
                  domain=int,
                  default=1)

    add_to_config("kf_xhat",
                  description="Resampling frequence for xhat (default 1)",
                  domain=int,
                  default=1)


def BM_config():
    # Bayraksan and Morton sequential (relative width)

    add_to_config("BM_h",
                  description="Controls width of confidence interval (default 1.75)",
                  domain=float,
                  default=1.75)

    add_to_config("BM_hprime",
                  description="Controls tradeoff between width and sample size (default 0.5)",
                  domain=float,
                  default=0.5)
    
    add_to_config("BM_eps",
                  description="Controls termination (default 0.2)",
                  domain=float,
                  default=0.2)
    
    add_to_config("BM_eps_prime",
                  description="Controls termination (default 0.1)",
                  domain=float,
                  default=0.1)
    
    add_to_config("BM_p",
                  description="Controls sample size (default 0.1)",
                  domain=float,
                  default=0.1)

    add_to_config("BM_q",
                  description="Related to sample size growth (default 1.2)",
                  domain=float,
                  default=1.2)


def BPL_config():
    # Bayraksan and Pierre-Louis

    add_to_config("BPL_eps",
                  description="Controls termination (default 1)",
                  domain=float,
                  default=1)

    add_to_config("BPL_c0",
                  description="Starting sample size (default 20)",
                  domain=int,
                  default=20)

    add_to_config("BPL_n0min",
                  description="Non-zero implies stochastic sampling (default 0)",
                  domain=int,
                  default=0)

