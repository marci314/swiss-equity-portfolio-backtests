# Data

Raw data is not included in this repository. Place these files directly in `data/`. Git ignores everything in this folder except this file.

| File | Needed by | Format | Source |
| --- | --- | --- | --- |
| `market_data.parquet` | use cases 03 and 04, all strategies, `ml_stock_selection/`, `report/build_report.py` | Daily panel with MultiIndex `(date, id)`; columns `price`, `mktcap`, `liquidity`, `sector` | Daily Swiss equity market data |
| `jkp_data.parquet` | same as above | Panel with MultiIndex `(date, id)`; 153 stock characteristic columns such as `ret_12_1`, `qmj`, `be_me` | Swiss stocks from the [JKP Global Factor Data](https://jkpfactors.com/) |
| `spi_index.csv` | same as above | First column is the date as `DD/MM/YYYY`, column `SPI` holds daily total returns as decimals | Swiss Performance Index |
| `IRSTCI01CHM156N.csv` | `ml_stock_selection/`, `report/build_report.py` | Columns `observation_date` (`YYYY-MM-DD`) and `IRSTCI01CHM156N` (monthly rate in percent per year) | [FRED series IRSTCI01CHM156N](https://fred.stlouisfed.org/series/IRSTCI01CHM156N), Swiss call-money rate |
| `msci_country_indices.csv` | use case 02 | First column `Index` is the date as `DD-MM-YYYY`, one column of daily returns per country | MSCI country indices |
| `NDDLWI.csv` | use case 02 | First column `Index` is the date as `DD-MM-YYYY`, column `NDDLWI` holds daily returns | MSCI World index series |
| `ml_signal_reg.parquet` | `strategies/ml_smart_beta_no_trade_band_sweep.ipynb` only | Precomputed ML rank predictions, one column per stock id | Generated separately; the file and the code that produced it are not part of this repository |

`use_cases/01_qp_solver_comparison_and_min_variance.ipynb` uses synthetic data and needs no files. `strategies/regime_aware_ml_allocator.ipynb` needs `market_data.parquet`, `jkp_data.parquet` and `spi_index.csv`.

## Generated results

The notebooks in `ml_stock_selection/` write results to `ml_stock_selection/sweep_results_*` and `ml_stock_selection/sweep_cache_*`. `report/build_report.py` reads these folders, and they are also ignored by git.
