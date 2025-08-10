import json
import os
import os.path
import re
import time
from pathlib import Path
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass

import h5py
import matplotlib
# Use non-interactive backend so figures are saved without popping up
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dataframes
import excell
import graph
import Key_functions as kf
import checkfunctions


@dataclass
class SaveOptions:
    save_negative_resistance: bool = True
    save_half_sweep: bool = True
    save_capacitive: bool = True
    save_let_through: bool = True


@dataclass
class PlotOptions:
    yield_vs_concentration: bool = True
    yield_vs_concentration_logx: bool = True
    yield_vs_spacing: bool = True
    yield_vs_spacing_logx: bool = False
    plot_3d: bool = True
    facet_by_polymer_conc_yield: bool = True
    facet_by_polymer_spacing_yield: bool = True
    correlation_heatmap: bool = True
    pairplot_numeric: bool = False
    violin_yield_by_polymer: bool = True
    box_yield_by_polymer: bool = True
    resistance_summary_plots: bool = True


@dataclass
class ClassificationOptions:
    filter_by_hdf5_labels: bool = True
    apply_capacitive_skip: bool = True
    apply_half_sweep_skip: bool = False
    apply_negative_resistance_skip: bool = False
    csv_source: str = "heuristic"  # or "hdf5"


def analyze_hdf5_levels(
    hdf5_file_path,
    solution_devices_excel_path,
    yield_dir_path,
    identifiers,
    match_all=True,
    voltage_val=0.1,
    output_dir="saved_files",
    valid_classifications=None,
    log_fn: Callable[[str], None] = print,
    plot_options: Optional[PlotOptions] = None,
    save_options: Optional[SaveOptions] = None,
    classification_options: Optional[ClassificationOptions] = None,
):
    start = time.time()

    if valid_classifications is None:
        valid_classifications = ["Memristive"]

    os.makedirs(output_dir, exist_ok=True)
    # Ensure subfolders used later exist
    for sub in [
        "negative_resistance",
        "half_sweep",
        "capacitive",
        "let_through",
    ]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    if plot_options is None:
        plot_options = PlotOptions()
    if save_options is None:
        save_options = SaveOptions()
    if classification_options is None:
        classification_options = ClassificationOptions()

    # get yield from file
    yield_device_dict, yield_device_sect_dict = get_yield_from_file(Path(yield_dir_path))

    with h5py.File(hdf5_file_path, "r") as store:
        # Group keys by depth

        grouped_keys = kf.filter_keys_by_identifiers(store, identifiers, match_all=match_all)
        # gets all keys
        all_keys = get_keys_at_depth(store, target_depth=5)

        list_of_substrate_names = collect_sample_names(grouped_keys)
        device_fabrication_info = get_fabrication_info(list_of_substrate_names, Path(solution_devices_excel_path))

        # current dataframes include
        # device fabrication info,
        # yield_device_dict, yield_device_sect_dict,
        # all_first_sweeps

        # Store the data on the first sweeps of all devices
        all_first_sweeps: List[Tuple[str, pd.DataFrame]] = []

        # Extract base keys by removing suffixes
        base_keys = sorted(set(k.rsplit('_', 2)[0] for k in grouped_keys))

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        # combine dataframes and dictionaries together for easier use
        df = dataframes.device_df(list_of_substrate_names, yield_device_dict, device_fabrication_info)
        log_fn(df)
        # Fill missing 'Yield' values with 0
        df['Yield'] = df['Yield'].fillna(0)
        log_fn("######################")
        log_fn(df)

        # summary and exploratory graphs
        plot_summary_graphs(df, identifiers, plot_options)

        # Analyze data for each key given
        for base_key in base_keys:
            # Retrieve both datasets at once
            df_raw_data, df_file_stats, parts_key = return_data(base_key, store)

            # store the first sweep of any device in a dataframe
            if parts_key[-1].startswith('1-'):
                all_first_sweeps.append((base_key, df_raw_data))

        initial_resistance(
            all_first_sweeps,
            voltage_val=voltage_val,
            valid_classifications=valid_classifications,
            output_dir=output_dir,
            log_fn=log_fn,
            save_options=save_options,
            classification_options=classification_options,
            plot_options=plot_options,
        )

        store.close()


def initial_resistance(
    data,
    voltage_val=0.1,
    valid_classifications=None,
    output_dir="saved_files",
    log_fn: Callable[[str], None] = print,
    save_options: Optional[SaveOptions] = None,
    classification_options: Optional[ClassificationOptions] = None,
    plot_options: Optional[PlotOptions] = None,
):
    """Find initial resistance between 0 and voltage_val and aggregate per device."""

    resistance_results = []
    wrong_classification = []

    if valid_classifications is None:
        valid_classifications = ["Memristive"]
    if save_options is None:
        save_options = SaveOptions()
    if classification_options is None:
        classification_options = ClassificationOptions()
    if plot_options is None:
        plot_options = PlotOptions()

    for key, value in data:
        # Extracting the relevant information from keys and generate safe keys
        safe_key = key.replace("/", "_")
        parts = key.strip('/').split('/')
        segments = parts[1].split("-")
        device_number = segments[0]
        polymer, polymer_percent = extract_polymer_info(segments[3])

        if classification_options.filter_by_hdf5_labels:
            try:
                classification = value['classification'].iloc[0]
            except Exception:
                classification = 'Unknown'
                log_fn(f"No classification found for key {key}")
            if classification not in valid_classifications:
                continue

        # Calculate resistance between the values of V
        resistance_data = value[(value['voltage'] >= 0) & (value['voltage'] <= voltage_val)]['resistance']
        resistance = resistance_data.mean()

        # various checks to remove bad data
        if resistance_data.empty:
            log_fn(f"No valid resistance data for {key}, skipping.")
            continue
        if resistance < 0:
            log_fn("check file as classification wrong - negative resistance seen on device")
            wrong_classification.append(key)
            if save_options.save_negative_resistance:
                label = (f" 0-0.1 {resistance}")
                fig = graph.plot_graph(value['voltage'], value['current'], "voltage", "current", label=label)
                fig.savefig(os.path.join(output_dir, f"negative_resistance/{safe_key}.png"))
                value.to_csv(os.path.join(output_dir, f"negative_resistance/{safe_key}.txt"), sep="\t")
                plt.close(fig)
            if classification_options.apply_negative_resistance_skip:
                # Skip adding to results
                continue

        # if not a full sweep between positive and negative
        if not ((value["current"].min() < 0) and (value["current"].max() > 0)):
            if save_options.save_half_sweep:
                fig = graph.plot_graph(value['voltage'], value['current'], "voltage", "current")
                fig.savefig(os.path.join(output_dir, f"half_sweep/{safe_key}.png"))
                plt.close(fig)
            if classification_options.apply_half_sweep_skip:
                continue

        capacitive = checkfunctions.is_sweep_capactive(value, key)
        if capacitive:
            log_fn(f"Device {device_number} is capacitive, skipping resistance calculation")
            if save_options.save_capacitive:
                fig = graph.plot_graph(value['voltage'], value['current'], "voltage", "current")
                fig.savefig(os.path.join(output_dir, f"capacitive/{safe_key}.png"))
                plt.close(fig)
            if classification_options.apply_capacitive_skip:
                continue
        else:
            log_fn(f"Calculated Average Resistance for key {key}: {resistance}")
            if save_options.save_let_through:
                fig = graph.plot_graph(value['voltage'], value['current'], "voltage", "current")
                fig.savefig(os.path.join(output_dir, f"let_through/{safe_key}.png"))
                plt.close(fig)

            # Store results
            resistance_results.append({
                'device_number': segments[0],
                'concentration': extract_concentration(segments[1]),
                'bottom_electrode': segments[2],
                'polymer': polymer,
                'polymer_percent': polymer_percent,
                'top_electrode': segments[4],
                'average_resistance': resistance,
                'classification': classification,
                'key': key
            })

    resistance_df = pd.DataFrame(resistance_results)

    # Group by device_number
    grouped = resistance_df.groupby('device_number')

    # Compute device statistics grouped
    device_stats = []
    for device, group in grouped:
        resistance = group['average_resistance'].mean()
        max_resistance = group['average_resistance'].max()
        min_resistance = group['average_resistance'].min()
        spread = (max_resistance - min_resistance) / 2

        log_fn(
            f"\nDevice {device}: Average Resistance: {resistance}, Max Resistance: {max_resistance}, Min Resistance: {min_resistance}, Spread: {spread}"
        )

        device_stats.append({
            'device_number': device,
            'average_resistance': resistance,
            'spread': spread
        })

    np.savetxt(os.path.join(output_dir, 'wrong_classifications.txt'), wrong_classification, fmt='%s')

    device_stats_df = pd.DataFrame(device_stats)
    device_stats_df.to_csv(os.path.join(output_dir, "Average_resistance_device_0.1v.csv"), index=False)
    resistance_df.to_csv(os.path.join(output_dir, "resistance_grouped_by_device_0.1v.csv"), index=False)

    # plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        device_stats_df['device_number'],
        device_stats_df['average_resistance'],
        yerr=device_stats_df['spread'],
        fmt='o',
        ecolor='red',
        capsize=5,
        label='Average Resistance with Error'
    )
    plt.xlabel('Device Number')
    plt.ylabel('Average Resistance (Ohms)')
    plt.title('Average Resistance by Device')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'device_average_resistance_with_error.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        resistance_df['device_number'],
        resistance_df['average_resistance'],
        fmt='x',
        ecolor='red',
        capsize=5,
        label='Average Resistance'
    )
    plt.xlabel('Device Number')
    plt.ylabel('Average Resistance (Ohms)')
    plt.title('Average Resistance by Device (individual sweeps)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'sweep_resistances_scatter.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        resistance_df['concentration'],
        resistance_df['average_resistance'],
        fmt='x',
        ecolor='red',
        capsize=5,
        label='Average Resistance'
    )
    plt.xlabel('Concentration')
    plt.ylabel('Average Resistance (Ohms)')
    plt.title('Resistance vs Concentration (first sweeps)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'resistance_vs_concentration.png'))
    plt.close()


def classify_sweeps(
    hdf5_file_path: str,
    identifiers: List[str],
    match_all: bool,
    output_dir: str = "saved_files",
    log_fn: Callable[[str], None] = print,
) -> str:
    """Produce a simple classification CSV from raw sweeps using heuristics.

    Returns path to the generated CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: List[dict] = []
    with h5py.File(hdf5_file_path, "r") as store:
        grouped_keys = kf.filter_keys_by_identifiers(store, identifiers, match_all=match_all)
        base_keys = sorted(set(k.rsplit('_', 2)[0] for k in grouped_keys))
        for base_key in base_keys:
            df_raw, _, parts = return_data(base_key, store)
            key = base_key
            classification = _classify_single_sweep(df_raw, key)
            results.append({
                "key": key,
                "classification": classification,
            })

    out_csv = os.path.join(output_dir, "classification_report.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    log_fn(f"Classification CSV written: {out_csv}")
    return out_csv


def _classify_single_sweep(df: pd.DataFrame, key: str) -> str:
    """Very simple heuristic-based classification placeholder.
    This can be replaced with a more advanced classifier in the future.
    """
    try:
        if checkfunctions.is_sweep_capactive(df, key):
            return "Capacitive"
        # Half-sweep check
        if not ((df["current"].min() < 0) and (df["current"].max() > 0)):
            return "Half-Sweep"
        # Negative resistance region heuristic
        sel = df[(df['voltage'] >= 0) & (df['voltage'] <= 0.1)]
        if not sel.empty and sel['resistance'].mean() < 0:
            return "Negative-Resistance"
        # Fallback
        return "Memristive"
    except Exception:
        return "Unknown"


def extract_concentration(concentration):
    match = re.search(r"[\d.]+", concentration)
    if match:
        return float(match.group())
    return None


def extract_polymer_info(polymer):
    name_match = re.match(r"[A-Za-z]+", polymer)
    percent_match = re.search(r"\((\d+)%\)", polymer)
    name = name_match.group() if name_match else None
    percentage = int(percent_match.group(1)) if percent_match else None
    return name, percentage


def get_keys_at_depth(store, target_depth=5):
    def traverse(group, current_depth, prefix=""):
        keys = []
        for name in group:
            path = f"{prefix}/{name}".strip("/")
            if isinstance(group[name], h5py.Group):
                keys.extend(traverse(group[name], current_depth + 1, path))
            elif isinstance(group[name], h5py.Dataset):
                if current_depth == target_depth:
                    keys.append(path)
        return keys

    return traverse(store, 1)


def get_yield_from_file(yield_path: Path):
    with open(yield_path / "yield_dict.json", "r") as f:
        sorted_yield_dict = json.load(f)

    with open(yield_path / "yield_dict_section.json", "r") as f:
        sorted_yield_dict_sect = json.load(f)

    return sorted_yield_dict, sorted_yield_dict_sect


def return_data(base_key, store):
    parts = base_key.strip('/').split('/')
    filename = parts[-1]
    device = parts[-2]
    section = parts[-3]

    key_file_stats = base_key + "_file_stats"
    key_raw_data = base_key + "_raw_data"

    data_file_stats = store[key_file_stats][()]
    data_raw_data = store[key_raw_data][()]

    column_names_raw_data = ['voltage', 'current', 'abs_current', 'resistance', 'voltage_ps', 'current_ps',
                             'voltage_ng', 'current_ng', 'log_Resistance', 'abs_Current_ps', 'abs_Current_ng',
                             'current_Density_ps', 'current_Density_ng', 'electric_field_ps', 'electric_field_ng',
                             'inverse_resistance_ps', 'inverse_resistance_ng', 'sqrt_Voltage_ps', 'sqrt_Voltage_ng',
                             'classification']

    column_names_file_stats = ['ps_area', 'ng_area', 'area', 'normalized_area', 'resistance_on_value',
                               'resistance_off_value', 'ON_OFF_Ratio', 'voltage_on_value', 'voltage_off_value']

    df_file_stats = pd.DataFrame(data_file_stats, columns=column_names_file_stats)
    df_raw_data_temp = pd.DataFrame(data_raw_data, columns=column_names_raw_data)
    df_raw_data = map_numbers_to_classification(df_raw_data_temp)

    return df_raw_data, df_file_stats, parts


def map_numbers_to_classification(df):
    if 'classification' in df.columns:
        reverse_classification_map = {
            0: 'Memristive',
            1: 'Capacitive',
            2: 'Conductive',
            3: 'Intermittent',
            4: 'Mem-Capacitance',
            5: 'Ohmic',
            6: 'Non-Conductive'
        }
        df['classification'] = df['classification'].map(reverse_classification_map)
    return df


def plot_summary_graphs(df, identifiers, plot_options: PlotOptions, base_dir=None):
    # prep the save location for graphs
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "Substrate_graphs")
    directory_name = ",".join(identifiers) if identifiers and identifiers != [""] else "All"

    save_loc = os.path.join(base_dir, directory_name)
    os.makedirs(save_loc, exist_ok=True)

    # Core scatter plots
    if plot_options.yield_vs_concentration:
        graph.plot_concentration_yield(df['Np Concentration'], df['Yield'], save_loc, directory_name)
    if plot_options.yield_vs_concentration_logx:
        graph.plot_concentration_yield(df['Np Concentration'], df['Yield'], save_loc, directory_name + " (logx)")
        plt.gca().set_xscale('log')
        plt.savefig(os.path.join(save_loc, 'concentration_yield_logx.png'))
        plt.close()
    if plot_options.yield_vs_spacing:
        graph.Spacing_yield(df['Qd Spacing (nm)'], df['Yield'], save_loc, directory_name)
    if plot_options.yield_vs_spacing_logx:
        graph.Spacing_yield(df['Qd Spacing (nm)'], df['Yield'], save_loc, directory_name + " (logx)")
        plt.gca().set_xscale('log')
        plt.savefig(os.path.join(save_loc, 'spacing_yield_logx.png'))
        plt.close()

    # Labeled versions
    graph.plot_concentration_yield_labels(df['Np Concentration'], df['Yield'], df["Device Name"], save_loc, directory_name)
    graph.Spacing_yield_labels(df['Qd Spacing (nm)'], df['Yield'], df["Device Name"], save_loc, directory_name)

    # 3D
    if plot_options.plot_3d:
        graph.Spacing_yield_concentration_3d(df["Np Concentration"], df["Qd Spacing (nm)"], df["Yield"],
                                             df["Device Name"], save_loc, directory_name)

    # Facets
    if plot_options.facet_by_polymer_conc_yield:
        graph.facet_concentration_yield_by_polymer(df, save_loc, directory_name)
    if plot_options.facet_by_polymer_spacing_yield:
        graph.facet_spacing_yield_by_polymer(df, save_loc, directory_name)

    # Correlations/Pairplots/Distributions
    if plot_options.correlation_heatmap:
        graph.correlation_heatmap(df, save_loc)
    if plot_options.pairplot_numeric:
        graph.pairplot_numeric(df, save_loc)
    if plot_options.violin_yield_by_polymer:
        graph.violin_yield_by_polymer(df, save_loc)
    if plot_options.box_yield_by_polymer:
        graph.box_yield_by_polymer(df, save_loc)


def collect_sample_names(grouped_keys):
    list_of_substrate_names = []
    current_sample = None
    for key in grouped_keys:
        parts = key.strip('/').split('/')
        substrate_name = parts[1]
        if substrate_name != current_sample:
            current_sample = substrate_name
            list_of_substrate_names.append(substrate_name)
    return list_of_substrate_names


def get_fabrication_info(list_of_substrate_names, solution_devices_excell_path: Path):
    device_fabrication_info = {}
    for name in list_of_substrate_names:
        df = excell.save_info_from_solution_devices_excell(name, solution_devices_excell_path)
        device_fabrication_info[name] = df
    return device_fabrication_info


