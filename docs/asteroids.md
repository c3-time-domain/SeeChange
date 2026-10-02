# Asteroid alert annotations

When building alerts, the checker predicts nearby known MPC asteroids at the
image midpoint and adds these fields to the current `diaSource`:

- `mpcDesignation`: the nearest designation within the radius, or null.
- `mpcMatches`: all neighbors, sorted by distance, with `designation` and `distanceArcsec`.
- `mpcMatchStatus`: `checked`, `disabled`, `unavailable`, or `stale`.

Only `checked` with an empty list means no neighbor was found. Matches and check
failures leave alert selection unchanged. Previous source entries are unchecked
and report `unavailable`. An angular neighbor is not a confirmed identification.

With the usual SeeChange dependencies and database configuration:

```sh
alembic upgrade head
python -m pipeline.refresh_asteroid_catalog --config /path/to/seechange_config.yaml
```

Schedule refreshes daily; use `--epoch 2025-12-18T00:00:00` for archival images.
Refreshes download and atomically replace the existing `mpc_table` orbit cache.
Each alert batch reads that cache once. Results are computed when alerts are
built and are not saved in the database.

Configure `asteroid_checking` in YAML: `enabled` (default true), `radius_arcsec`
(30), `max_epoch_distance_days` (3), `prefilter_padding_deg` (1), and
`observatory_code` (null). Set the instrument's MPC ground-observatory code for
topocentric positions; null uses Earth's center. UTC image midpoints are converted
to TDB. A padded two-body footprint search is refined with n-body propagation and
light-time correction. The prefilter is approximate; its completeness and
positional accuracy still need validation against independent ephemerides.

Run the focused tests in the `asteroids` environment with SeeChange dependencies:

```sh
conda activate asteroids
export PYTHONPATH="$PWD/extern/RBbot_inference${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
from util.config import Config
Config.init('default_config.yaml', setdefault=True)
import pytest
raise SystemExit(pytest.main([
    '--noconftest', '-q',
    'tests/pipeline/test_asteroid_checker.py',
    'tests/pipeline/test_asteroid_annotations.py',
]))
PY
```

These tests cover matching, synthetic kete predictions, alert construction,
failure statuses, and Avro compatibility without the full pipeline services.
