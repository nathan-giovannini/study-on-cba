# SPDX-License-Identifier: MIT
"""
Add manually defined offshore hubs to a PyPSA-Eur network.

The hubs are described by three CSV files in data/offshore/ (one file per
PyPSA component):

buses.csv        one row per bus
    name         bus name. Offshore hubs are added to the network as new buses;
                 onshore buses use ENTSO-E market node names (NOS0, DKW1, DE00,
                 UK00, UKNI, ...) and are only placeholders (see below)
    x, y         lon, lat
    country      country the bus belongs to (ISO code, e.g. DE, GB; Northern
                 Ireland = GB). It is set on the offshore buses added to the
                 network and decides which hubs are imported (see below)
    offshore     True  -> offshore hub bus
                 False -> placeholder for an existing onshore bus
    carrier, v_nom   optional (defaults: AC, 380 kV)

links.csv        one row per link
    name, bus0, bus1   bus names as defined in buses.csv
    p_nom              MW; leave EMPTY to make the link extendable
    carrier, p_nom_extendable, p_nom_max, p_min_pu, p_max_pu, efficiency,
    capital_cost       optional

generators.csv   one row per generator (attached to an offshore hub)
    name, bus, carrier
    p_nom              MW; leave EMPTY to make the generator extendable
    p_nom_extendable, p_nom_max   optional

Countries
---------
A country is "present in the model" if the network has at least one onshore AC
bus with that country code. An offshore hub is imported only if its `country`
is present. A link is imported only if both of its buses are imported (so a
link between a hub and an onshore bus of a country that is not in the model is
skipped), and a generator only if its hub is imported. Everything that is
skipped is written to the log.

Onshore buses
-------------
An onshore bus of buses.csv (e.g. "DE00") is NOT added to the network. Each
link end that refers to one is mapped to the existing onshore bus of the same
`country` that is closest to the offshore bus at the other end of the link.
If an onshore bus already carries the name of a bus of the network, that bus
is used as is. The mapping is written to the log.

Generator availability is taken from the atlite cutout at the hub location,
using the config["renewable"] settings of the generator carrier
(e.g. offwind-ac, offwind-dc). Any p_max_pu column in generators.csv is ignored.
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

BUS_COLS = ["name", "x", "y", "country", "offshore"]
LINK_COLS = ["name", "bus0", "bus1", "p_nom"]
GEN_COLS = ["name", "bus", "carrier", "p_nom"]

# optional link attributes that are passed on to PyPSA if present in links.csv
LINK_ATTRS = [
    "carrier",
    "p_nom",
    "p_nom_max",
    "p_min_pu",
    "p_max_pu",
    "efficiency",
    "capital_cost",
]


# --------------------------------------------------------------------------
# reading / validation
# --------------------------------------------------------------------------
def _to_bool(s, default=False):
    return s.map(
        lambda v: default
        if pd.isna(v)
        else str(v).strip().lower() in {"true", "1", "yes", "y"}
    ).astype(bool)


def _read_csv(fn, required):
    df = pd.read_csv(fn, comment="#", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{fn}: missing column(s) {missing}")
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()
    dup = df.loc[df["name"].duplicated(), "name"].tolist()
    if dup:
        raise ValueError(f"{fn}: duplicated names {dup}")
    return df.set_index("name")


def _is_extendable(row, columns):
    """Explicit p_nom_extendable column wins; otherwise empty p_nom = extendable."""
    if "p_nom_extendable" in columns and not pd.isna(row["p_nom_extendable"]):
        return bool(row["p_nom_extendable"])
    return bool(pd.isna(row["p_nom"]))


def read_offshore(buses_fn, links_fn, gens_fn):
    buses = _read_csv(buses_fn, BUS_COLS)
    links = _read_csv(links_fn, LINK_COLS)
    gens = _read_csv(gens_fn, GEN_COLS)

    buses["offshore"] = _to_bool(buses["offshore"])
    if buses["country"].isna().any():
        raise ValueError(
            f"buses.csv: empty country for {buses.index[buses['country'].isna()].tolist()}"
        )
    if "v_nom" not in buses:
        buses["v_nom"] = 380.0
    if "carrier" not in buses:
        buses["carrier"] = "AC"

    ends = pd.concat([links["bus0"], links["bus1"]]).unique()
    unknown = sorted(set(ends) - set(buses.index))
    if unknown:
        raise ValueError(f"links.csv refers to buses not in buses.csv: {unknown}")

    both_onshore = links[
        ~buses.loc[links["bus0"], "offshore"].values
        & ~buses.loc[links["bus1"], "offshore"].values
    ]
    if len(both_onshore):
        raise ValueError(
            f"links with two onshore ends are not supported: {both_onshore.index.tolist()}"
        )

    unknown = sorted(set(gens["bus"]) - set(buses.index))
    if unknown:
        raise ValueError(f"generators.csv refers to buses not in buses.csv: {unknown}")
    onshore_gens = gens.index[~buses.loc[gens["bus"], "offshore"].values].tolist()
    if onshore_gens:
        raise ValueError(f"generators attached to onshore buses: {onshore_gens}")

    return buses, links, gens


# --------------------------------------------------------------------------
# keep only what belongs to countries of the model
# --------------------------------------------------------------------------
def _onshore_ac(n):
    country = n.buses["country"].fillna("").astype(str)
    return n.buses[(n.buses["carrier"] == "AC") & (country != "")]


def filter_by_country(n, buses, links, gens):
    """Drop hubs of countries that are not in the network and everything
    attached to them (links, generators)."""
    present = set(_onshore_ac(n)["country"].astype(str))
    logger.info(f"Countries present in the model: {sorted(present)}")

    in_network = buses.index.isin(n.buses.index)
    keep = buses["country"].astype(str).isin(present) | (~buses["offshore"] & in_network)

    dropped_hubs = buses.index[~keep & buses["offshore"]].tolist()
    if dropped_hubs:
        logger.info(f"Skipping offshore bus(es) of absent countries: {dropped_hubs}")

    kept_links = keep.loc[links["bus0"]].values & keep.loc[links["bus1"]].values
    dropped_links = links.index[~kept_links].tolist()
    if dropped_links:
        logger.info(f"Skipping link(s) with an end outside the model: {dropped_links}")

    kept_gens = keep.loc[gens["bus"]].values
    dropped_gens = gens.index[~kept_gens].tolist()
    if dropped_gens:
        logger.info(f"Skipping generator(s) of absent countries: {dropped_gens}")

    buses, links, gens = buses[keep], links[kept_links], gens[kept_gens]

    linked = set(links["bus0"]) | set(links["bus1"])
    for hub in buses.index[buses["offshore"]]:
        if hub not in linked:
            logger.warning(f"Offshore bus '{hub}' has no link left; its generation is stranded.")
    return buses, links, gens


# --------------------------------------------------------------------------
# mapping of onshore placeholders to buses of the network
# --------------------------------------------------------------------------
def _haversine_km(x0, y0, x1, y1):
    x0, y0, x1, y1 = map(np.radians, (x0, y0, x1, y1))
    a = np.sin((y1 - y0) / 2) ** 2 + np.cos(y0) * np.cos(y1) * np.sin((x1 - x0) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def make_resolver(n, buses):
    """Return f(bus, other) -> network bus name.

    Must be called BEFORE the hubs are added, so that only buses of the
    original network are candidates.
    """
    onshore = _onshore_ac(n)
    original = set(n.buses.index)
    log = {}

    def resolve(bus, other):
        if buses.at[bus, "offshore"]:
            return bus
        if bus in original:
            return bus
        key = (bus, other)
        if key not in log:
            cand = onshore[
                onshore["country"].astype(str) == str(buses.at[bus, "country"])
            ]
            if cand.empty:
                raise ValueError(
                    f"No onshore bus with country '{buses.at[bus, 'country']}' "
                    f"in the network for bus '{bus}'."
                )
            d = _haversine_km(
                cand["x"].values,
                cand["y"].values,
                buses.at[other, "x"],
                buses.at[other, "y"],
            )
            i = int(np.argmin(d))
            log[key] = cand.index[i]
            logger.info(
                f"Onshore bus '{bus}' (towards '{other}') -> '{cand.index[i]}' "
                f"({d[i]:.0f} km)"
            )
        return log[key]

    return resolve


# --------------------------------------------------------------------------
# atlite profiles
# --------------------------------------------------------------------------
def capacity_factor(cutout, params):
    """Capacity-factor field (time, y, x) of the whole cutout for one carrier.

    `params` is config["renewable"][carrier]; it is read the same way as in
    scripts/build_renewable_profiles.py.
    """
    resource = dict(params["resource"])
    resource.pop("method", None)  # always "wind"

    # turbine can be a single model or a {year: model} dict -> take the first
    models = resource["turbine"]
    if isinstance(models, dict):
        resource["turbine"] = models[next(iter(models))]

    return cutout.wind(capacity_factor_timeseries=True, **resource)


def profile_at(cf, x, y, params, snapshots):
    """Time series of the cutout grid cell nearest to (x, y)."""
    dx = float(abs(cf.x[1] - cf.x[0])) if cf.sizes["x"] > 1 else 0.0
    dy = float(abs(cf.y[1] - cf.y[0])) if cf.sizes["y"] > 1 else 0.0
    if not (
        float(cf.x.min()) - dx <= x <= float(cf.x.max()) + dx
        and float(cf.y.min()) - dy <= y <= float(cf.y.max()) + dy
    ):
        logger.warning(f"Location ({x}, {y}) lies outside the cutout; using nearest cell.")

    correction = params.get("correction_factor", 1.0)  # top level, not in resource
    ts = correction * cf.sel(x=x, y=y, method="nearest").to_series()

    clip = params.get("clip_p_max_pu")
    if clip is not None:
        ts = ts.where(ts >= clip, 0.0)

    return ts.reindex(snapshots)


# --------------------------------------------------------------------------
# adding the components
# --------------------------------------------------------------------------
def add_offshore(n, buses, links, gens, cutout, renewable):
    resolve = make_resolver(n, buses)  # before any bus is added

    # reference costs from the existing generators, taken BEFORE adding the new
    # ones so that they do not bias the mean
    ref_cost = {}
    for c in gens["carrier"].unique():
        ref = n.generators[n.generators["carrier"] == c]
        ref_cost[c] = (
            (ref["marginal_cost"].mean(), ref["capital_cost"].mean())
            if len(ref)
            else (0.0, 0.0)
        )

    carriers = {"DC"} | set(gens["carrier"]) | set(buses["carrier"])
    if "carrier" in links:
        carriers |= set(links["carrier"].dropna())
    for c in carriers:
        if c not in n.carriers.index:
            n.add("Carrier", c)

    # ---- offshore buses ----
    for name, b in buses[buses["offshore"]].iterrows():
        n.add("Bus", name, carrier=b["carrier"])
        n.buses.loc[name, "x"] = b["x"]
        n.buses.loc[name, "y"] = b["y"]
        n.buses.loc[name, "v_nom"] = float(b["v_nom"])
        n.buses.loc[name, "country"] = b["country"]
        n.buses.loc[name, "sub_network"] = "99"

    # ---- links ----
    for name, l in links.iterrows():
        kw = {a: l[a] for a in LINK_ATTRS if a in links.columns and not pd.isna(l[a])}
        ext = _is_extendable(l, links.columns)
        kw["carrier"] = kw.get("carrier", "DC")
        kw["p_nom"] = 0.0 if ext else float(l["p_nom"])
        kw["p_nom_extendable"] = ext
        if not ext:
            kw["capital_cost"] = 0.0
        n.add(
            "Link",
            name,
            bus0=resolve(l["bus0"], l["bus1"]),
            bus1=resolve(l["bus1"], l["bus0"]),
            **kw,
        )

    # ---- generators ----
    cf_cache, profile_cache = {}, {}
    for name, g in gens.iterrows():
        carrier = g["carrier"]
        if carrier not in renewable:
            raise KeyError(
                f"Generator '{name}': carrier '{carrier}' is not in config['renewable']."
            )
        params = renewable[carrier]
        if carrier not in cf_cache:
            cf_cache[carrier] = capacity_factor(cutout, params)
        key = (g["bus"], carrier)
        if key not in profile_cache:
            profile_cache[key] = profile_at(
                cf_cache[carrier],
                buses.at[g["bus"], "x"],
                buses.at[g["bus"], "y"],
                params,
                n.snapshots,
            )

        ext = _is_extendable(g, gens.columns)
        p_nom_max = g["p_nom_max"] if "p_nom_max" in gens.columns else np.nan
        marginal_cost, capital_cost = ref_cost[carrier]
        n.add(
            "Generator",
            name,
            bus=g["bus"],
            carrier=carrier,
            p_nom=0.0 if ext else float(g["p_nom"]),
            p_nom_extendable=ext,
            p_nom_max=np.inf if pd.isna(p_nom_max) else float(p_nom_max),
            p_max_pu=profile_cache[key],
            marginal_cost=marginal_cost,
            capital_cost=capital_cost if ext else 0.0,
        )

    logger.info(
        f"Added {int(buses['offshore'].sum())} offshore bus(es), "
        f"{len(links)} link(s) and {len(gens)} generator(s)."
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("add_offshore_hubs", clusters=39, opts="")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)
    buses, links, gens = read_offshore(
        snakemake.input.buses, snakemake.input.links, snakemake.input.generators
    )
    buses, links, gens = filter_by_country(n, buses, links, gens)

    sns = get_snapshots(snakemake.params.snapshots, snakemake.params.drop_leap_day)
    cutout = load_cutout(snakemake.input.cutout, time=sns)

    add_offshore(n, buses, links, gens, cutout, snakemake.params.renewable)
    n.export_to_netcdf(snakemake.output.network)
