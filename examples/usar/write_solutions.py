import os

import matplotlib.pyplot as plt
import pyomo.environ as pyo

from plot import plot_walks, plot_gantt


def walks_writer(
    walks_dir: str, scen_name: str, scen_mod: pyo.ConcreteModel, bundling: bool
) -> None:
    plot_walks(scen_mod)
    plt.savefig(os.path.join(walks_dir, scen_name + ".pdf"))
    plt.close()


def gantt_writer(
    gantt_dir: str, scen_name: str, scen_mod: pyo.ConcreteModel, bundling: bool
) -> None:
    plot_gantt(scen_mod, scen_name)
    plt.savefig(os.path.join(gantt_dir, scen_name + ".pdf"))
    plt.close()
