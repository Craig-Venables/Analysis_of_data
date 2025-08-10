import argparse
import json
import os
import os.path
import re
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openpyxl as op  # noqa: F401 - imported for side-effects/compatibility
import pandas as pd
import scipy.stats as stats  # noqa: F401 - currently unused in core flow
import warnings

import dataframes
import excell
import graph
import Key_functions as kf
import checkfunctions
from analysis import analyze_hdf5_levels

# add lines on graph showing where short circuiting resistance is,

# if running into issue with yield run other code first plotting all the graphs!

# Remove warning for excell file
warnings.simplefilter("ignore")

# haver a look at gpt on building an index instead of looping though all keys all the time

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Analyze memristor HDF5 data and generate plots/statistics.")
    parser.add_argument("--hdf5", dest="hdf5_file", required=False, help="Path to the HDF5 file to analyze")
    parser.add_argument("--excel", dest="excel_path", required=False, help="Path to 'solutions and devices.xlsx'")
    parser.add_argument("--yield-dir", dest="yield_dir", required=False, help="Directory containing yield_dict.json and yield_dict_section.json")
    parser.add_argument(
        "-i", "--identifier", dest="identifiers", action="append", default=[], help="Identifier to filter keys (can be repeated)"
    )
    parser.add_argument("--match-any", dest="match_all", action="store_false", help="Match any identifier instead of all (default: match all)")
    parser.add_argument("--voltage", dest="voltage_val", type=float, default=0.1, help="Voltage upper bound for resistance calculation (default: 0.1)")
    parser.add_argument(
        "--class", dest="valid_classes", action="append", default=None, help="Valid classifications to include (can be repeated). Default: Memristive"
    )
    parser.add_argument("--output-dir", dest="output_dir", default=str(Path.cwd() / "saved_files"), help="Directory to save outputs (default: repo/saved_files)")
    parser.add_argument("--gui", dest="gui", action="store_true", help="Launch the PyQt GUI instead of CLI")
    parser.add_argument("--classify", dest="run_classify", action="store_true", help="Run classification-only flow and export classification_report.csv")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)

    if args.gui:
        # Lazy import to avoid PyQt dependency in CLI-only environments
        from gui import launch_gui

        launch_gui()
        return 0

    # Provide sensible defaults if not all CLI params were passed (non-GUI mode)
    hdf5_fallback, excel_fallback, yield_fallback = _discover_defaults()
    hdf5_path = args.hdf5_file or hdf5_fallback
    excel_path = args.excel_path or excel_fallback
    yield_dir = args.yield_dir or yield_fallback

    # Validate required paths
    if not hdf5_path or not excel_path or not yield_dir:
        print("Error: --hdf5, --excel, and --yield-dir are required (or discoverable). Or use --gui.")
        return 2

    identifiers = args.identifiers if args.identifiers else ["PMMA"]
    valid_classes = args.valid_classes if args.valid_classes is not None else ["Memristive"]

    if args.run_classify:
        from analysis import classify_sweeps
        classify_sweeps(
            hdf5_file_path=hdf5_path,
            identifiers=identifiers,
            match_all=args.match_all,
            output_dir=args.output_dir,
            log_fn=print,
        )
        else:
        analyze_hdf5_levels(
            hdf5_file_path=hdf5_path,
            solution_devices_excel_path=excel_path,
            yield_dir_path=yield_dir,
            identifiers=identifiers,
            match_all=args.match_all,
            voltage_val=args.voltage_val,
            output_dir=args.output_dir,
            valid_classifications=valid_classes,
            log_fn=print,
        )

    return 0





def _discover_defaults():
    """Try to auto-discover paths for hdf5, excel, and yield dir on this machine.
    Returns (hdf5_path, excel_path, yield_dir) where missing ones are None.
    """
    # Base root the user specified
    base = Path(r"C:\Users\Craig-Desktop\OneDrive - The University of Nottingham\Documents\Phd\2) Data\1) Devices\1) Memristors")
    hdf5_candidates = []
    if base.exists():
        for p in base.glob("Memristor_data_*.h5"):
            hdf5_candidates.append(p)
    hdf5_path = None
    if hdf5_candidates:
        # pick latest by stem suffix date (digits at end)
        def _key(p: Path):
            # Extract 8-digit date from filename if present
            s = p.stem
            digits = ''.join(ch for ch in s if ch.isdigit())
            return digits
        hdf5_path = str(sorted(hdf5_candidates, key=_key)[-1])

    # Guess excel and yield
    excel_guess = Path.home() / Path("OneDrive - The University of Nottingham/Documents/Phd/solutions and devices.xlsx")
    excel_path = str(excel_guess) if excel_guess.exists() else None

    yield_guess = Path.home() / Path("OneDrive - The University of Nottingham/\Documents/Phd/1) Projects/1) Memristors/1) Curated Data")
    yield_dir = str(yield_guess) if yield_guess.exists() else None

    return hdf5_path, excel_path, yield_dir

if __name__ == "__main__":
    raise SystemExit(main())