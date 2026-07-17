# Data Card: UCI Bike Sharing Dataset

## Dataset Name

Bike Sharing Dataset

## Source

UCI Machine Learning Repository

- **URL:** https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- **DOI:** 10.24432/C5W894
- **License:** CC BY 4.0

## Citation

Fanaee-T, Hadi, and Gama, Joao. "Event labeling combining ensemble detectors
and background knowledge." Progress in Artificial Intelligence (2013):
pp. 1-15, Springer Berlin Heidelberg.

## Description

The dataset contains hourly counts of rental bikes between 2011-01-01 and
2012-12-31 in the Capital Bikeshare system, Washington, D.C., USA.

## Schema

### hour.csv (primary dataset)

| Column | Type | Description |
|---|---|---|
| instant | int | Record index |
| dteday | date | Date |
| season | int (1-4) | 1=spring, 2=summer, 3=fall, 4=winter |
| yr | int (0-1) | Year (0=2011, 1=2012) |
| mnth | int (1-12) | Month |
| hr | int (0-23) | Hour |
| holiday | int (0/1) | Whether the day is a holiday |
| weekday | int (0-6) | Day of the week |
| workingday | int (0/1) | 1 if neither weekend nor holiday |
| weathersit | int (1-4) | Weather situation (1=clear, 2=cloudy, 3=light rain, 4=heavy rain) |
| temp | float | Normalised temperature (Celsius / 41) |
| atemp | float | Normalised feeling temperature |
| hum | float | Normalised humidity |
| windspeed | float | Normalised wind speed |
| casual | int | Count of casual users |
| registered | int | Count of registered users |
| cnt | int | Total rental count (casual + registered) |

### day.csv

Daily aggregated version of the same data. Not used in this project.

## License

CC BY 4.0. Attribution required when publishing results derived from this
dataset.

## Known Biases

- Data covers only two years from a single city (Washington, D.C.).
- Weather measurements are from a single airport station.
- The system and ridership patterns may have changed since 2011-2012.

## Usage Restrictions

The dataset is publicly available under CC BY 4.0. No restrictions on use.
