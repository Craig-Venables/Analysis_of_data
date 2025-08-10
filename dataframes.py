import pandas as pd
def device_df(list_of_substrate_names,yield_device_dict,device_fabrication_info):
    data = []

    for name in list_of_substrate_names:
        # Get yield from the dictionary
        yield_ = yield_device_dict.get(name, "Key not found")

        # Access device fabrication info
        file = device_fabrication_info.get(name)

        if file is not None:
            # Extract required fields
            qd_spacing = file.get("Qd Spacing (nm)", "Key not found")
            qd_concentration = file.get('Np Concentraion')
            qd_volume_fraction = file.get('Volume Fraction')
            qd_volume_fraction_percent = file.get('Volume Fraction %')
            qd_weight_fract = file.get('Weight Fraction')
            solution_id = file.get('Solution 1 ID')
            polymer = file.get('Polymer')

            # If concentration is 'Stock', store as 0
            if qd_concentration == ' Stock':
                #print(qd_spacing, "appending 0")
                qd_concentration = 0

            # Append the data as a dictionary (row)
            data.append({
                "Device Name": name,
                "Yield": yield_,
                "Qd Spacing (nm)": qd_spacing,
                "Np Concentration": qd_concentration,
                "Polymer": polymer,
                "Volume Fraction": qd_volume_fraction,
                "Volume Fraction %": qd_volume_fraction_percent,
                "Weight Fraction": qd_weight_fract,
                "Solution ID": solution_id
            })
        else:
            print(f"Warning: Device '{name}' not found in 'Fabrication Info'.")
    df = pd.DataFrame(data)
    return df