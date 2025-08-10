## Analysis of Memristor Data

This project analyzes memristor HDF5 datasets, generates plots/statistics, and relates them to fabrication metadata stored in an Excel workbook. It now supports both a command-line interface (CLI) and a GUI built with PyQt5.

### Features
- Filter HDF5 dataset keys by identifiers (e.g., polymer types) and analyze first sweeps
- Compute average initial resistance over a voltage window
- Aggregate per-device stats and export to CSV
- Generate summary plots and categorized sweep images
- Read fabrication info from `solutions and devices.xlsx`
- Read yield info from `yield_dict.json` and `yield_dict_section.json`
- Run via CLI or GUI

### Installation
1. Create and activate a virtual environment (recommended)
2. Install dependencies:
```
pip install -r requirements.txt
```

### Data Requirements
- An HDF5 file with datasets named like `<...>/<device>/<section>/<filename>_raw_data` and `<...>_file_stats`
- Excel workbook: `solutions and devices.xlsx` with sheets:
  - `Memristor Devices`
  - `Devices Overview`
  - `Prepared Solutions`
- Yield directory containing:
  - `yield_dict.json`
  - `yield_dict_section.json`

### CLI Usage
Run analysis without GUI:
```
python main.py --hdf5 PATH/TO/data.h5 --excel PATH/TO/solutions and devices.xlsx --yield-dir PATH/TO/yield_dir -i PMMA --voltage 0.1 --output-dir saved_files
```

Options:
- `--hdf5`: HDF5 file path (required)
- `--excel`: Excel workbook path (required)
- `--yield-dir`: Directory with yield JSON files (required)
- `-i/--identifier`: Filter identifier; repeatable (default: PMMA)
- `--match-any`: Match any identifier instead of all
- `--voltage`: Voltage upper bound for resistance calc (default: 0.1)
- `--class`: Valid classifications to include; repeatable (default: Memristive)
- `--output-dir`: Where to save outputs (default: repo `saved_files/`)
- `--gui`: Launch GUI instead of CLI
- `--classify`: Run classification-only and write `classification_report.csv`

### GUI Usage
```
python main.py --gui
```
Then use the file pickers and options, and click Run. Tooltips explain each option. The GUI auto-detects:
- latest HDF5 at `C:\Users\Craig-Desktop\OneDrive - The University of Nottingham\Documents\Phd\2) Data\1) Devices\1) Memristors` matching `Memristor_data_YYYYMMDD.h5`
- Excel/yield JSONs under your OneDrive PhD documents
Outputs go to a `saved_files/` folder inside this repo by default.

### Output
- Plots: `1.png`, `2.png`, `3.png` under output dir
- CSVs: `Average_resistance_device_0.1v.csv`, `resistance_grouped_by_device_0.1v.csv`
- Categorized sweep images under `negative_resistance/`, `half_sweep/`, `capacitive/`, `let_through/`

### Recommended Upgrades
- Packaging: convert this repo into an installable package; add a console script entrypoint
- Config: support a `yaml`/`toml` config file for default paths and identifiers
- Logging: switch prints to the `logging` module with file/console handlers
- Tests: add unit tests (e.g., for key parsing, resistance calc) with sample HDF5 fixtures
- Performance: index HDF5 datasets and vectorize scans; optionally parallelize per-device analysis
- Robustness: add schema/version validation for HDF5 and Excel inputs
- Visualization: add toggles for log scales, facet plots by polymer/percent, export SVG, violin plots of resistance spreads, and device-level boxplots
- Data Quality: add outlier detection (IQR/Z-score) as optional filters
- CLI/GUI parity: expose more options in GUI (e.g., choose which plots to generate)

### Calling from other Python code
Import and call `analysis.analyze_hdf5_levels(...)` or `analysis.classify_sweeps(...)` for integration into a larger toolbox.


