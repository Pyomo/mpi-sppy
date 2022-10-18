"""Provides a function which is called after optimization."""
import pyomo.environ as pyo


def scenario_denouement(
    rank: int, name: str, scenario: pyo.ConcreteModel
) -> None:
    """Does nothing (is a no-op).

    This function is called after optimization finishes.

    Args:
        rank: Unused.
        name: Unused.
        scenario: Unused.
    """
    pass
