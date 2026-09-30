# SPDX-License-Identifier: MIT
"""
Add manually defined offshore hubs to a PyPSA-Eur network.

Each hub = 1 bus + N generators + 1 link to an onshore bus.
Inputs come from data/offshore_hubs.csv (one row per generator).

CSV columns
-----------
hub                 hub name (rows with the same name form one hub)
x, y                hub location (lon, lat) - repeated on every row of the hub
bus1                onshore bus the hub connects to - repeated
link_p_nom          link capacity in MW; leave EMPTY to make it extendable - repeated
link_capital_cost   EUR/MW/a, only relevant if the link is extendable - repeated
gen                 generator label (unique within the hub)
carrier             generator carrier (e.g. offwind-dc)
p_nom               MW; leave EMPTY to make the generator extendable
p_nom_max           MW upper bound for extendable generators (empty = inf)
"""

import logging

import numpy as np
import pandas as pd
import pypsa

from scripts._helpers import (
    configure_logging,
    get_snapshots,
    load_cutout,
    set_scenario_config,
)

logger = logging.getLogger(__name__)

HUB_COLS = ["x", "y", "bus1", "link1_p_nom", "link1_capital_cost", "bus2", "link2_p_nom", "link2_capital_cost"]


def read_hubs(fn):
    df = pd.read_csv(fn, comment="#", skipinitialspace=True)
    for hub, grp in df.groupby("hub"):
        if len(grp[HUB_COLS].drop_duplicates()) > 1:
            # NaN != NaN, so re-check ignoring NaNs before complaining
            if len(grp[HUB_COLS].fillna(-1).drop_duplicates()) > 1:
                raise ValueError(f"Hub '{hub}' has inconsistent hub-level columns.")
    return df


def atlite_profile(cutout, x, y, params, snapshots):
    """Capacity-factor time series of the cutout grid cell nearest to (x, y).

    `params` is config["renewable"]["offwind-dc"]; it is read the same way
    as in scripts/build_renewable_profiles.py.
    """
    resource = dict(params["resource"])
    resource.pop("method", None)  # always "wind"

    # turbine can be a single model or a {year: model} dict -> take the first
    models = resource["turbine"]
    if isinstance(models, dict):
        resource["turbine"] = models[next(iter(models))]

    correction = params.get("correction_factor", 1.0)  # top level, not in resource

    cf = cutout.wind(capacity_factor_timeseries=True, **resource)  # (time, y, x)
    ts = correction * cf.sel(x=x, y=y, method="nearest").to_series()

    clip = params.get("clip_p_max_pu")
    if clip is not None:
        ts = ts.where(ts >= clip, 0.0)

    return ts.reindex(snapshots)


def add_hubs(n, df, cutout, params):
    for c in ["DC"] + df["carrier"].unique().tolist():
        if c not in n.carriers.index:
            n.add("Carrier", c)

    for hub, grp in df.groupby("hub", sort=False):
        h = grp.iloc[0]
        n.add("Bus", hub)
        n.buses.loc[hub, ["x", "y", "sub_network", "country", "v_nom"]] = [h.x, h.y, "99","", 380.0]

        # ---- link to shore: no length, bidirectional ----
        ext = pd.isna(h.link1_p_nom)
        n.add(
            "Link",
            f"{hub} to {h.bus1}",
            bus0=hub,
            bus1=h.bus1,
            carrier="DC",
            p_nom=0 if ext else h.link1_p_nom,
            p_nom_extendable=bool(ext),
            p_min_pu=-1,
            efficiency=1.0,
            capital_cost=0 if pd.isna(h.link1_capital_cost) else h.link1_capital_cost,
        )

        n.add(
            "Link",
            f"{hub} to {h.bus2}",
            bus0=hub,
            bus1=h.bus2,
            carrier="DC",
            p_nom=0 if ext else h.link2_p_nom,
            p_nom_extendable=bool(ext),
            p_min_pu=-1,
            efficiency=1.0,
            capital_cost=0 if pd.isna(h.link2_capital_cost) else h.link2_capital_cost,
        )
        # ---- atlite profile at the hub location (one per hub) ----
        profile = atlite_profile(cutout, h.x, h.y, params, n.snapshots)

        # cost defaults copied from the existing offshore generators of that carrier
        for _, g in grp.iterrows():
            ref = n.generators.query("carrier == @g.carrier")
            gext = pd.isna(g.p_nom)
            n.add(
                "Generator",
                f"{hub} {g.gen} {g.carrier}",
                bus=hub,
                carrier=g.carrier,
                p_nom=0 if gext else g.p_nom,
                p_nom_extendable=bool(gext),
                p_nom_max=np.inf if pd.isna(g.p_nom_max) else g.p_nom_max,
                p_max_pu=profile,
                marginal_cost=ref.marginal_cost.mean() if len(ref) else 0.0,
                capital_cost=(ref.capital_cost.mean() if len(ref) else 0.0)
                if gext
                else 0.0,
            )
        logger.info(f"Added hub {hub} with {len(grp)} generator(s).")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("add_offshore_hubs", clusters=39, opts="")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)
    df = read_hubs(snakemake.input.hubs)

    sns = get_snapshots(snakemake.params.snapshots, snakemake.params.drop_leap_day)
    cutout = load_cutout(snakemake.input.cutout, time=sns)

    add_hubs(n, df, cutout, snakemake.params.renewable["offwind-dc"])
    n.export_to_netcdf(snakemake.output.network)