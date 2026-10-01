# Known-asteroid alert annotations

The pipeline checks measured detections after scoring and before saving and
sending alerts. Asteroid matches do not modify measurement quality, scores,
object association, or alert selection. Both matched and unmatched detections
continue through the existing alert rules.

## Alert fields

The current `diaSource` includes:

- `mpcDesignation`: the nearest known asteroid within the configured radius,
  or null, retained for compatibility with the existing field.
- `mpcMatches`: all known asteroid neighbors within the radius, sorted by
  angular distance. Each entry contains `designation` and `distanceArcsec`.
- `mpcMatchStatus`: `checked`, `disabled`, `unavailable`, or `stale`.
- `mpcCatalogVersion`: the orbit snapshot version used for the annotation.
- `mpcCatalogEpoch`: the snapshot epoch, as a TDB Julian Date.

An empty neighbor list means no known asteroid was found only when the status
is `checked`. `unavailable` includes an unpopulated catalog or a failed check;
`stale` means the image time is outside the allowed propagation window. These
statuses do not stop alert sending. A positional match is a proximity annotation,
not a confirmed moving-object identification, and unknown asteroids are not covered.

Readers should use the updated `share/avsc/ls4.v0_2.diaSource.avsc` schema.
New fields have defaults so existing packets can still be read with the updated
schema. Previous source entries without saved annotations report `unavailable`.

## Setup and refresh

Run in the SeeChange application environment, with the usual database and
storage configuration and installed requirements (including `kete==2.1.5`):

```sh
export SEECHANGE_CONFIG=/absolute/path/to/seechange_config.yaml
alembic upgrade head
python -m pipeline.refresh_asteroid_catalog --config "$SEECHANGE_CONFIG"
```

Schedule that refresh command daily using the deployment's job scheduler.
It downloads known MPC orbits, propagates them to the requested epoch, and
atomically replaces the catalog and its version metadata. A failed refresh
rolls back the replacement. Image processing never downloads the orbit catalog.
Workers check the version before each image and reload states when it changes.

For archival processing, populate a snapshot near the image date:

```sh
python -m pipeline.refresh_asteroid_catalog --config "$SEECHANGE_CONFIG" \
  --epoch 2025-12-18T00:00:00
```

Configuration defaults:

```yaml
asteroid_checking:
  enabled: true
  radius_arcsec: 30.0
  max_epoch_distance_days: 3.0
  prefilter_padding_deg: 1.0
  observatory_code: null
```

Set `observatory_code` to the instrument's MPC ground-observatory code for
observer parallax. The null default uses Earth's center. The exposure midpoint
is converted from UTC MJD to TDB JD. A fast, light-time-corrected two-body pass
selects a padded cone around the actual image footprint; an n-body pass refines
candidate positions, followed by a short two-body light-time correction.
The padding protects against approximation errors but is not a guarantee for
all orbit uncertainties or very close encounters; tune and validate it against
representative observations before deployment.

Annotations are saved separately in `asteroid_match_sets`, keyed by measurement
UUID and checking provenance. Reordering or filtering measurements does not
shift matches. Saved successful checks retain the original snapshot version on
reruns. Unavailable and stale results are retried when processing is rerun.

## Validation

The focused tests exercise neighbor matching, real `kete` predictions,
serialization, unchanged Kafka selection, and failure statuses. Database tests
use a fresh temporary schema and test actual migrations, atomic refresh rollback,
worker reload, and datastore save/reload. They skip unless a PostgreSQL DSN is
provided; never use the full repository's database fixtures against a production
database.

To run these tests without the repository's external pipeline fixtures, use
an environment with the SeeChange dependencies and initialize the bundled source
packages if they are not already installed:

```sh
git submodule update --init extern/nersc-upload-connector extern/RBbot_inference
export PYTHONPATH="$PWD/extern/RBbot_inference${PYTHONPATH:+:$PYTHONPATH}"
export SEECHANGE_ASTEROID_TEST_DSN='host=localhost port=5432 dbname=testdb user=testuser'
python - <<'PY'
from util.config import Config
Config.init('default_config.yaml', setdefault=True)
import pytest
raise SystemExit(pytest.main([
    '--noconftest', '-q',
    'tests/pipeline/test_asteroid_checker.py',
    'tests/pipeline/test_asteroid_annotations.py',
    'tests/pipeline/test_asteroid_database.py',
]))
PY
```

The database tests create only the upstream schema needed by these checks.
A full exposure-to-alert run still requires the ordinary SeeChange services,
reference images, calibration data, and broker configuration.
