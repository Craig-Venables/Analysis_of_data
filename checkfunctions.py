import numpy as np

def is_sweep_capactive(df,key):
    current_at_zero_voltage = df.loc[df['voltage'] == 0, 'current']
    max_current = df['current'].max()
    #print("++")
    #print(key)
    #print("max_current" ,max_current)
    #print("max_current" ,max_current /1000)
    #print(current_at_zero_voltage.iloc[1])
    #print("++")
    # print(current_at_zero_voltage)
    # print(current_at_zero_voltage[1])
    if current_at_zero_voltage.iloc[1] > max_current /1000:
        if current_at_zero_voltage.iloc[1] <= 1E-12:
            return True
        else:
            return False

    else:
        return False

def detect_large_fluctuations(value, threshold=1):
    """Detects whether the dataset has large fluctuations in current"""
    current = value['current'].values
    voltage = value['voltage'].values

    # Compute absolute differences between consecutive points
    diff_current = np.abs(np.diff(current))
    diff_voltage = np.abs(np.diff(voltage))

    # Avoid division by zero
    safe_mean = np.mean(np.abs(current)) if np.mean(np.abs(current)) > 0 else 1

    # Normalized fluctuation measure
    fluctuation_score = np.mean(diff_current) / safe_mean
    fluctuation_std = np.std(diff_current) / safe_mean

    # If fluctuations exceed threshold, reject the dataset
    if fluctuation_score > threshold or fluctuation_std > threshold:
        print(
            f"Dataset rejected due to large fluctuations (score={fluctuation_score:.2f}, std={fluctuation_std:.2f})")
        return True
    return False

# # Use Z-score to remove outliers
# Dosnt work
# z_scores = np.abs(stats.zscore(value["resistance"]))
# if (z_scores >= 5).any():
#     print(f"Device {device_number} yields-score is too high, skipping")
#     fig = graph.plot_graph(value['voltage'], value['current'], "voltage", "current")
#     fig.savefig(f"saved_files/z_score_less_3/{safe_key}.png")  # Save with corrected filename
#     plt.close(fig)  # Close the figure to free memory
#     continue
# (4) **Outlier detection using IQR**
# Q1 = resistance_data.quantile(0.25)
# Q3 = resistance_data.quantile(0.75)
# IQR = Q3 - Q1
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
#
# if (resistance < lower_bound) or (resistance > upper_bound):
#     print(f"Device {device_number} resistance is an outlier (IQR filtering). Skipping.")
#     fig = graph.plot_graph(value['voltage'], value['current'], "voltage", "current")
#     fig.savefig(f"saved_files/iqr/{safe_key}.png")  # Save with corrected filename
#     plt.close(fig)  # Close the figure to free memory
#     continue


# Convert to numpy array
# resistance_series = resistance_data.to_numpy()

# **Check excessive fluctuations**
# if detect_large_fluctuations(value):
#     print(f"Skipping {key} due to excessive fluctuations.")
#     fig = graph.plot_graph(value['voltage'], value['current'], "voltage", "current")
#     fig.savefig(f"saved_files/ef/{safe_key}.png")  # Save with corrected filename
#     plt.close(fig)  # Close the figure to free memory
#     continue
# df_filtered = value[z_scores < 3]  # Keep only values within 3 standard deviations