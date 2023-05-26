"""Provides functions to plot the solution of a scenario."""
import os

import matplotlib.pyplot as plt
import pyomo.environ as pyo

from plot import plot_walks, plot_gantt


def walks_writer(
    walks_dir: str, scen_name: str, scen_mod: pyo.ConcreteModel, bundling: bool
) -> None:
    """Writes a geographical plot of rescue team movements to a PDF.

    Args:
        walks_dir: Path of directory in which plots are saved.
        scen_name: The scenario name for titling the plot.
        scen_mod: Solved USAR `ConcreteModel` for the scenario.
        bundling: Unused.
    """
    plot_walks(scen_mod, scen_name)
    plt.savefig(os.path.join(walks_dir, scen_name + ".pdf"))
    plt.close()


def gantt_writer(
    gantt_dir: str, scen_name: str, scen_mod: pyo.ConcreteModel, bundling: bool
) -> None:
    """Writes a Gantt chart of rescue team schedules to a PDF.

    Args:
        gantt_dir: Path of directory in which plots are saved.
        scen_name: The scenario name for titling the plot.
        scen_mod: Solved USAR `ConcreteModel` for the scenario.
        bundling: Unused.
    """
    plot_gantt(scen_mod, scen_name)
    plt.savefig(os.path.join(gantt_dir, scen_name + ".pdf"))
    plt.close()
