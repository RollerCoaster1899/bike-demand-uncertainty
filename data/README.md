# Data

## Source

UCI Bike Sharing dataset (dataset id 275).

- **Original source:** UCI Machine Learning Repository
- **DOI:** 10.24432/C5W894
- **License:** CC BY 4.0
- **URL:** https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip
- **Records:** 17,389 hourly records from 2011-01-01 to 2012-12-31

## Raw Data

Download with:

```bash
uv run python scripts/download_data.py
```

The zip file is extracted to `data/raw/` and contains:

- `hour.csv` -- Hourly rental data (the primary dataset)
- `day.csv` -- Daily aggregated data (not used)

## Processed Data

The pipeline creates an aligned hourly index with no missing timestamps
before computing lagged features. Feature-incomplete rows are dropped.

## Synthetic Fixture

For CI and smoke testing, a deterministic synthetic fixture is generated
in code. It is clearly labeled as synthetic and not used for real evaluation.
