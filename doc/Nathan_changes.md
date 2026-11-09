# Changes compared with the mainline

This summary covers the changes on `feature/tyndp-load-profile` relative to the
pull request's base branch, `pypsa-eur-v08-2026`.

## TYNDP electricity demand

- Adds `load.source` to select OPSD (the default) or TYNDP demand, plus
  `load.tyndp_scenario` (`NT`, `DE`, or `GA`) and
  `load.available_years_tyndp`.
- Extends TYNDP retrieval to unpack `Demand-Profiles.zip` and adds a cleaning
  rule that reads the selected scenario and planning year. Only the first
  configured planning horizon is used.
- Adds the TYNDP demand profile to the electricity-demand build. It is
  reindexed to the snapshots' year without gap filling and assigned directly
  to matching TYNDP buses; unmatched nodes are dropped with a warning.
- Adds the `safe_pyear` helper for falling back to an available planning year.
- Updates the load validation model, default configuration, JSON schema, and
  `config/test/config.tyndp.yaml` to exercise the TYNDP source.

## Other branch changes

- Adds a collection rule for composed networks with offshore hubs and a sample
  `data/offshore_hubs.csv`.
- Adds two custom configurations under `config/nathan_config/` for TYNDP-network
  and demand runs.
- Ignores files under `/benchmark-tyndp/`.
- Removes `config/scenarios.template.yaml`.
