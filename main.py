import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import h5py
import time
import h5py
import os.path
from pathlib import Path
import excell
import sys
import json
import openpyxl as op
import warnings

import graph
import checkfunctions
# l

# Remove warning for excell file
warnings.simplefilter("ignore")

# filepaths
user_dir = Path.home()
hdf5_file = user_dir / Path(
    "OneDrive - The University of Nottingham/Documents/Phd/1) Projects/1) Memristors/4) Code analysis/memristor_data.h5")
solution_devices_excell_path = user_dir / Path(
    "OneDrive - The University of Nottingham/Documents/Phd/solutions and devices.xlsx")
yield_path = user_dir / Path(
    "OneDrive - The University of Nottingham\Documents/Phd/1) Projects/1) Memristors/1) Curated Data")



# haver a look at gpt on buidling an index instead of looping though all keys all the time


def analyze_hdf5_levels(hdf5_file):
    start = time.time()
    # a list of all substrate names
    list_of_substrate_names = []
    # list by substrate name of all fabrication information
    device_fabrication_info = {}
    # get yield from file
    yield_device_dict, yield_device_sect_dict = get_yield_from_file(yield_path)

    with h5py.File(hdf5_file, "r") as store:
        # Group keys by depth
        grouped_keys = get_keys_at_depth(store, target_depth=5)
        print(grouped_keys)

        # collect all substrate names
        current_sample = None
        for key in grouped_keys:
            parts = key.strip('/').split('/')
            substrate_name = parts[1]
            if substrate_name != current_sample:
                current_sample = substrate_name
                list_of_substrate_names.append(substrate_name)

        # get all fabrication info for each substrate,
        for name in list_of_substrate_names:
            df = excell.save_info_from_solution_devices_excell(name, solution_devices_excell_path)
            device_fabrication_info[name] = df  # Store in dictionary with name as key
            # todo this needs to have a key of the device name

        print(device_fabrication_info["D26-Stock-ITO-F8PFB(1%)-Gold-s4"])
        # print(device_fabrication_info[list_of_substrate_names[0]])

        # Store the data on the first sweeps of all devices
        all_first_sweeps = []
        all_second_sweeps = []
        all_third_sweeps = []
        all_four_sweeps = []
        all_five_sweeps = []

        # Extract base keys by removing suffixes and co
        base_keys = sorted(set(k.rsplit('_', 2)[0] for k in grouped_keys))

        # Analyze data for each file!
        for base_key in base_keys:
            print(base_key)

            # Retrieve both datasets at once
            df_raw_data, df_file_stats, parts_key = return_data(base_key, store)
            # parts_key = filename(parts[-1]) , device(parts[-2]) etc...

            # use here to do any extra processing

            # store the first five sweeps of any device in a dataframe
            if parts_key[-1].startswith('1-'):
                all_first_sweeps.append((base_key, df_raw_data))
            if parts_key[-1].startswith('2-'):
                all_second_sweeps.append((base_key, df_raw_data))
            if parts_key[-1].startswith('3-'):
                all_third_sweeps.append((base_key, df_raw_data))
            if parts_key[-1].startswith('4-'):
                all_four_sweeps.append((base_key, df_raw_data))
            if parts_key[-1].startswith('5-'):
                all_five_sweeps.append((base_key, df_raw_data))

        #print(all_first_sweeps)
        # First sweep data
        initial_resistance(all_first_sweeps)
        store.close()

    #print("time to organise the data before calling inisital first sweep ", middle - start)


def initial_resistance(data, voltage_val=0.1):
    """ Finds the initial reseistance between 0-0.1 V for the list of values given
        also filters for data that's not within the list valid_classifications to remove unwanted data
    """

    resistance_results = []
    wrong_classification = []

    # Define valid classifications
    #valid_classifications = ["Memristive", "Ohmic", "Conductive", "intermittent","Mem-Capacitance"]
    valid_classifications = ["Memristive"]

    for key, value in data:
        """value = all the data (metrics_df)
            key = folder structure"""

        # Extracting the relevant information from keys and generate safe keys
        safe_key = key.replace("/", "_")
        parts = key.strip('/').split('/')
        segments = parts[1].split("-")
        device_number = segments[0]
        polymer, polymer_percent = extract_polymer_info(segments[3])

        try:
            classification = value['classification'].iloc[0]
        except:
            classification = 'Unknown'
            print(f"No classification found for key {key}")

        if classification in valid_classifications:


            # Calculate resistance between the values of V
            resistance_data = value[(value['voltage'] >= 0) & (value['voltage'] <= voltage_val)]['resistance']
            resistance = resistance_data.mean()

            # calculate gradient of line for the data to see difference

            capacitive = checkfunctions.is_sweep_capactive(value, key)

            # various checks to remove bad data

            if resistance < 0:
                print("check file as classification wrong - negative resistance seen on device")
                # print(key)
                wrong_classification.append(key)

                # saves all images rejected for later verification
                label = (f" 0-0.1 {resistance}")
                fig = graph.plot_graph(value['voltage'], value['current'], "voltage", "current",label=label)
                fig.savefig(f"saved_files/negative_resistance/{safe_key}.png")  # Save with corrected filename
                value.to_csv(f"saved_files/negative_resistance/{safe_key}.txt", sep ="\t")
                plt.close(fig)  # Close the figure to free memory

            if (value['current'].min() > 0) or (value['current'].max() < 0):
                fig = graph.plot_graph(value['voltage'], value['current'], "voltage", "current")
                fig.savefig(f"saved_files/half_sweep/{safe_key}.png")  # Save with corrected filename
                plt.close(fig)  # Close the figure to free memory

            if capacitive:
                print(f"Device {device_number} is capacitive, skipping resistance calculation")
                safe_key = key.replace("/", "_")
                fig = graph.plot_graph(value['voltage'], value['current'], "voltage", "current")
                fig.savefig(f"saved_files/capacitive/{safe_key}.png")  # Save with corrected filename
                plt.close(fig)  # Close the figure to free memory

                # maybe if resistance is <0 it should pull the second sweep until a
                # value is found as sometimes the first sweeps non_conductive?

            else:
                # Print calculated resistance for debugging
                print(f"Calculated Average Resistance for key {key}: {resistance}")

                # Store results for checking later
                fig = graph.plot_graph(value['voltage'], value['current'], "voltage", "current")
                fig.savefig(f"saved_files/let_through/{safe_key}.png")  # Save with corrected filename
                plt.close(fig)  # Close the figure to free memory


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

    # also plot graph of all the resistances seen within a device and not average them like below

    # Compute device statistics grouped
    device_stats = []
    for device, group in grouped:
        resistance = group['average_resistance'].mean()
        max_resistance = group['average_resistance'].max()
        min_resistance = group['average_resistance'].min()
        spread = (max_resistance - min_resistance) / 2

        print(f"\nDevice {device}: Average Resistance: {resistance}, Max Resistance: {max_resistance}, "
              f"Min Resistance: {min_resistance}, Spread: {spread}")

        device_stats.append({
            'device_number': device,
            'average_resistance': resistance,
            'spread': spread
        })

    np.savetxt('wrong_classifications.txt', wrong_classification, fmt='%s')

    device_stats_df = pd.DataFrame(device_stats)
    device_stats_df.to_csv("saved_files/Average_resistance_device_0.1v.csv", index=False)
    resistance_df.to_csv("saved_files/resistance_grouped_by_device_0.1v.csv", index=False)

    # plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        device_stats_df['device_number'],  # x values
        device_stats_df['average_resistance'],  # y values
        yerr=device_stats_df['spread'],  # Error bars
        fmt='o',
        ecolor='red',
        capsize=5,
        label='Average Resistance with Error'
    )
    plt.xlabel('Device Number')
    plt.ylabel('Average Resistance (Ohms)')
    plt.title('Average Resistance by Device')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('saved_files/1.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        resistance_df['device_number'],  # x values
        resistance_df['average_resistance'],  # y values
        fmt='x',
        ecolor='red',
        capsize=5,
        label='Average Resistance'
    )
    plt.xlabel('Device Number')
    plt.ylabel('Average Resistance (Ohms)')
    plt.title('Average Resistance by Device  non averaged')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('saved_files/2.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        resistance_df['concentration'],  # x values
        resistance_df['average_resistance'],  # y values
        fmt='x',
        ecolor='red',
        capsize=5,
        label='Average Resistance'
    )
    plt.xlabel('concentration')
    plt.ylabel('Average Resistance (Ohms)')
    plt.title('Average Resistance by Device  non averaged')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('saved_files/3.png')
    plt.show()


def extract_concentration(concentration):
    match = re.search(r"[\d.]+", concentration)  # Match numbers and decimal points
    if match:
        return float(match.group())  # Convert to float
    return None


def extract_polymer_info(polymer):
    name_match = re.match(r"[A-Za-z]+", polymer)  # Match only letters at the start
    percent_match = re.search(r"\((\d+)%\)", polymer)  # Match the percentage in parentheses
    name = name_match.group() if name_match else None
    percentage = int(percent_match.group(1)) if percent_match else None
    return name, percentage


def get_keys_at_depth(store, target_depth=5):
    """
    Recursively traverse the HDF5 file and return keys at the specified depth.

    Parameters:
    - store: h5py.File or h5py.Group object
    - target_depth: int, depth at which to collect keys

    Returns:
    - List of keys at the specified depth
    """

    def traverse(group, current_depth, prefix=""):
        keys = []
        for name in group:
            path = f"{prefix}/{name}".strip("/")
            if isinstance(group[name], h5py.Group):  # If it's a group, recurse
                keys.extend(traverse(group[name], current_depth + 1, path))
            elif isinstance(group[name], h5py.Dataset):  # If it's a dataset, check depth
                if current_depth == target_depth:
                    keys.append(path)
        return keys

    return traverse(store, 1)  # Start at depth 1


def get_yield_from_file(yield_path):
    """
    Load and return the yield dictionary from the JSON files.

    Parameters:
    - yield_path: Path object to the yield directory"""

    # Load the JSON files
    with open(yield_path / "yield_dict.json", "r") as f:
        sorted_yield_dict = json.load(f)

    with open(yield_path / "yield_dict_section.json", "r") as f:
        sorted_yield_dict_sect = json.load(f)

    print(sorted_yield_dict)
    print(sorted_yield_dict["D26-Stock-ITO-F8PFB(1%)-Gold-s4"])
    return sorted_yield_dict, sorted_yield_dict_sect


def return_data(base_key, store):
    """
    Given the file key return the data in a pd dataframe converting form numpy array
    """
    parts = base_key.strip('/').split('/')
    print(parts)
    filename = parts[-1]
    device = parts[-2]
    section = parts[-3]

    key_file_stats = base_key + "_file_stats"
    key_raw_data = base_key + "_raw_data"

    data_file_stats = store[key_file_stats][()]
    data_raw_data = store[key_raw_data][()]

    # convert data back to pd dataframe

    column_names_raw_data = ['voltage', 'current', 'abs_current', 'resistance', 'voltage_ps', 'current_ps',
                             'voltage_ng', 'current_ng', 'log_Resistance', 'abs_Current_ps', 'abs_Current_ng',
                             'current_Density_ps', 'current_Density_ng', 'electric_field_ps', 'electric_field_ng',
                             'inverse_resistance_ps', 'inverse_resistance_ng', 'sqrt_Voltage_ps', 'sqrt_Voltage_ng',
                             'classification']

    column_names_file_stats = ['ps_area', 'ng_area', 'area', 'normalized_area', 'resistance_on_value',
                               'resistance_off_value', 'ON_OFF_Ratio', 'voltage_on_value', 'voltage_off_value']

    # Convert numpy arrays to pd dataframes
    df_file_stats = pd.DataFrame(data_file_stats, columns=column_names_file_stats)
    df_raw_data_temp = pd.DataFrame(data_raw_data, columns=column_names_raw_data)
    df_raw_data = map_numbers_to_classification(df_raw_data_temp)

    # print(df_file_stats)
    # print(df_raw_data)

    return df_raw_data, df_file_stats, parts


def map_numbers_to_classification(df):
    # Only apply the mapping if the 'classification' column exists in the dataframe
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


def filter_keys_by_suffix(keys, suffix):
    """
    Filter keys by a specific suffix (e.g., '_info', '_metrics').
    """
    return [key for key in keys if key.endswith(suffix)]


# Run analysis on _metrics data
analyze_hdf5_levels(hdf5_file)
