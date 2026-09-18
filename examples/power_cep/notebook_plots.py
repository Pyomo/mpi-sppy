###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2026, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.pyplot as plt  # noqa: F401
    import pandas as pd


def _normalize_rows(table):
    if hasattr(table, "to_dict"):
        return table.to_dict(orient="records")
    return list(table)


TECH_COLORS = {
    "gas": "#4C78A8",
    "wind": "#54A24B",
    "solar": "#F2C14E",
    "load_shedding": "#D62728",
}

def _autopct_mw(values):
    total = sum(values) if values else 0.0

    def _fmt(pct):
        mw = total * pct / 100.0
        return f"{mw:.0f} MW"

    return _fmt


def plot_existing_system_pie(ax, existing_df: pd.DataFrame):
    rows = _normalize_rows(existing_df)
    values = [row["capacity_mw"] for row in rows]
    ax.pie(
        values,
        labels=[row["unit"] for row in rows],
        autopct=_autopct_mw(values),
        startangle=90,
    )
    ax.set_title("Existing system by unit")


def plot_demand_by_scenario(
    ax,
    demand_df: pd.DataFrame,
    hours: list[str],
    scenario_labels: list[str] | None = None,
):
    rows = _normalize_rows(demand_df)
    scenarios = list(dict.fromkeys(row["scenario"] for row in rows))
    width = 0.8 / max(len(scenarios), 1)
    x = range(len(hours))
    for idx, scen in enumerate(scenarios):
        df = {row["hour"]: row for row in rows if row["scenario"] == scen}
        offsets = [i + (idx - (len(scenarios) - 1) / 2) * width for i in x]
        label = scenario_labels[idx] if scenario_labels and idx < len(scenario_labels) else scen
        ax.bar(offsets, [df.get(hour, {}).get("demand_mw", 0.0) for hour in hours], width=width, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(hours, rotation=30, ha="right")
    ax.set_ylabel("Demand (MW)")
    ax.set_title("Representative-hour demand by scenario")
    ax.legend()


def plot_build_pies(ax_left, ax_right, ev_builds_mw: dict[str, float], rp_builds_mw: dict[str, float]):
    labels = ["gas", "wind", "solar"]
    ev_vals = [float(ev_builds_mw.get(k, 0.0)) for k in labels]
    rp_vals = [float(rp_builds_mw.get(k, 0.0)) for k in labels]
    colors = [TECH_COLORS[k] for k in labels]

    ax_left.pie(ev_vals, labels=labels, colors=colors, autopct=_autopct_mw(ev_vals), startangle=90)
    ax_left.set_title("EV new builds")

    ax_right.pie(rp_vals, labels=labels, colors=colors, autopct=_autopct_mw(rp_vals), startangle=90)
    ax_right.set_title("RP new builds")


def plot_generation_side_by_side(axs, generation_df: pd.DataFrame, hours: list[str], scenario_order: list[str], label: str):
    rows = _normalize_rows(generation_df)
    techs = ["gas", "wind", "solar"]
    for ax, scen in zip(axs, scenario_order):
        df = {row["hour"]: row for row in rows if row["scenario"] == scen}
        x = range(len(hours))
        bottom = [0.0] * len(hours)
        for tech in techs:
            vals = [df.get(hour, {}).get(tech, 0.0) for hour in hours]
            ax.bar(x, vals, bottom=bottom, color=TECH_COLORS[tech], label=tech if scen == scenario_order[0] else None)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.bar(x, [df.get(hour, {}).get("load_shedding", 0.0) for hour in hours], bottom=bottom, color=TECH_COLORS["load_shedding"], label="load_shedding" if scen == scenario_order[0] else None)
        ax.set_xticks(list(x))
        ax.set_xticklabels(hours, rotation=30, ha="right")
        ax.set_title(f"{label} — {scen}")
        ax.set_ylabel("MW")


def plot_generation_comparison(rp_axes, eev_axes, rp_df: pd.DataFrame, eev_df: pd.DataFrame, hours: list[str], scenario_order: list[str]):
    plot_generation_side_by_side(rp_axes, rp_df, hours, scenario_order, "RP")
    plot_generation_side_by_side(eev_axes, eev_df, hours, scenario_order, "EEV")


def plot_generation_comparison_grid(
    ax_grid,
    rp_df: pd.DataFrame,
    eev_df: pd.DataFrame,
    hours: list[str],
    scenario_order: list[str],
    scenario_labels: list[str] | None = None,
):
    rp_rows = _normalize_rows(rp_df)
    eev_rows = _normalize_rows(eev_df)
    labels = scenario_labels if scenario_labels is not None else scenario_order

    for row_idx, scen in enumerate(scenario_order):
        for col_idx, (rows, label) in enumerate(((rp_rows, "RP"), (eev_rows, "EEV"))):
            ax = ax_grid[row_idx, col_idx]
            df = {row["hour"]: row for row in rows if row["scenario"] == scen}
            x = range(len(hours))
            bottom = [0.0] * len(hours)
            for tech in ["gas", "wind", "solar", "load_shedding"]:
                vals = [df.get(hour, {}).get(tech, 0.0) for hour in hours]
                ax.bar(
                    x,
                    vals,
                    bottom=bottom,
                    color=TECH_COLORS[tech],
                    label=tech if row_idx == 0 and col_idx == 0 else None,
                )
                bottom = [b + v for b, v in zip(bottom, vals)]
            ax.set_xticks(list(x))
            ax.set_xticklabels(hours, rotation=30, ha="right")
            ax.set_title(f"{label} — {labels[row_idx]}")
            ax.set_ylabel("MW")


def plot_generation_comparison_figure(
    fig,
    ax_grid,
    rp_df: pd.DataFrame,
    eev_df: pd.DataFrame,
    hours: list[str],
    scenario_order: list[str],
    scenario_labels: list[str] | None = None,
):
    plot_generation_comparison_grid(ax_grid, rp_df, eev_df, hours, scenario_order, scenario_labels)
    add_shared_legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))


def _stacked_generation_plot(ax, df: pd.DataFrame, hours: list[str], title: str):
    bottom = [0.0] * len(hours)
    x = range(len(hours))
    for tech in ["gas", "wind", "solar", "load_shedding"]:
        vals = df.set_index("hour").reindex(hours)[tech].fillna(0.0).tolist()
        ax.bar(x, vals, bottom=bottom, color=TECH_COLORS[tech], label=tech if title.endswith("RP") else None)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(hours, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("MW")


def plot_generation_rows_by_scenario(axs, gen_df: pd.DataFrame, hours: list[str], scenario_order: list[str], title_prefix: str):
    for ax, scen in zip(axs, scenario_order):
        df = gen_df[gen_df["scenario"] == scen]
        _stacked_generation_plot(ax, df, hours, f"{title_prefix} — {scen}")


def add_shared_legend(fig):
    import matplotlib.pyplot as plt

    handles, labels = [], []
    for tech in ["gas", "wind", "solar", "load_shedding"]:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=TECH_COLORS[tech]))
        labels.append(tech)
    fig.legend(handles, labels, loc="lower center", ncol=4)
