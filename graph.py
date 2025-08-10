import matplotlib
# Use non-interactive backend for saving-only in background threads
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os


def plot_graph(x_data, y_data, xlabel="x", ylabel="y", label="Undefined", xscale="", yscale="", yerr=None):
    fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axis
    if yerr is not None:
        ax.errorbar(x_data, y_data, yerr=yerr, fmt='-o', ecolor='red', capsize=5, label=label)  # Line with error bars
    else:
        ax.plot(x_data, y_data, '-o', label=label)  # Line graph with points
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Average Resistance by Device')

    if yscale:
        ax.set_yscale(yscale)

    ax.legend()
    ax.grid(True)
    
    return fig  # Return the figure obje


def Spacing_yield_concentration_3d(x, y, yields, labels,Save_loc,directory_name):
    """ 3d graph """

    # Convert to NumPy arrays
    x = np.array(x)
    y = np.array(y)
    yields = np.array(yields)
    labels = np.array(labels)

    # Remove zero values from y (and corresponding x, yields, labels)
    mask = yields != 0
    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = yields[mask]
    labels_filtered = labels[mask]

    # Apply log transformation to x values
    x_log = np.log10(x_filtered)  # Apply log transformation here
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the scatter plot in 3D
    ax.scatter(x_filtered, y_filtered, z_filtered, c='r')

    # for adding labels
    # # Label each point with the device name and add an offset to avoid overlap
    # for i in range(len(x_filtered)):
    #     offset_x = np.random.uniform(-0.1, 0.1)
    #     offset_y = np.random.uniform(-0.1, 0.1)
    #     offset_z = np.random.uniform(-0.1, 0.1)
    #     ax.text(x_filtered[i] + offset_x, y_filtered[i] + offset_y, z_filtered[i] + offset_z,
    #             labels_filtered[i], fontsize=8)

    # Set axis labels and title

    #ax.set_xscale('log')


    # possible to fix ticks for log this way
    # plt.xticks([-0.69897, -1.15490196],
    #            ["0.2", "0.07"])

    ax.set_xlabel('Concentration')
    ax.set_ylabel('Spacing (nm)')
    ax.set_zlabel('Yield')

    ax.set_title("3D Plot of Spacing, Yield, and Concentration "+ directory_name )



    plt.savefig(Save_loc + '/3d_spacing_yield_concenration.png')


def plot_generic_graph_labels(x, y, labels, xlabel="x", ylabel="y", title="Undefined", xscale="", yscale="",
                              concentration=False):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if not yscale == "":
        plt.yscale()
    if not xscale == "":
        plt.xscale()

    if concentration:
        #plt.xscale('log')
        plt.xticks([0.001, 0.005, 0.05, 1, 0.07, 0.2, 2, 4, 0.4, 0.1, 0.01],
                   ["0.001", "0.005", "0.05", " 1", "0.07", "0.2", "2", "4", "0.4", "0.1", "0.01"])

    # Extract the first part of each label before the "-"
    labels_filtered = [label.split('-')[0] for label in labels]

    # Add labels to each point
    for i in range(len(x)):
        plt.text(x[i], y[i], labels_filtered[i], fontsize=8, ha='right', va='bottom')




def plot_concentration_yield(x, y,Save_loc,directory_name):
    x = pd.to_numeric(pd.Series(x), errors='coerce')
    y = pd.to_numeric(pd.Series(y), errors='coerce')
    mask = x.notna() & y.notna()
    plt.figure(figsize=(15, 10))
    plt.scatter(x[mask], y[mask])
    plt.xlabel("Concentration ")
    plt.ylabel("Yield")
    plt.title("Concentration Vs Yield " + directory_name)

    #plt.xscale('log')
    plt.xticks([0.001, 0.005, 0.05, 1, 0.07, 0.2, 2, 4, 0.4, 0.1],
               ["0.001", "0.005", "0.05", " 1", "0.07", "0.2", "2", "4", "0.4", "0.1"])

    plt.savefig(Save_loc + '/concentration_yield.png')



def plot_concentration_yield_labels(x, y, labels,Save_loc,directory_name):
    x = pd.to_numeric(pd.Series(x), errors='coerce')
    y = pd.to_numeric(pd.Series(y), errors='coerce')
    labels = list(labels)
    mask = (x.notna() & y.notna()).values
    x_f = x[mask]
    y_f = y[mask]
    labels_f = [labels[i] for i, m in enumerate(mask) if m]
    plt.figure(figsize=(15, 10))
    plt.scatter(x_f, y_f)
    plt.xlabel("Concentration ")
    plt.ylabel("Yield")
    plt.title("Concentration Vs Yield " + directory_name )

    #plt.xscale('log')
    plt.xticks([0.001, 0.005, 0.05, 1, 0.07, 0.2, 2, 4, 0.4, 0.1, 0.01],
               ["0.001", "0.005", "0.05", " 1", "0.07", "0.2", "2", "4", "0.4", "0.1", "0.01"])

    # Extract the first part of each label before the "-"
    labels_filtered = [label.split('-')[0] for label in labels_f]

    # Add labels to each point
    for i in range(len(x_f)):
        plt.text(x_f.iloc[i], y_f.iloc[i], labels_filtered[i], fontsize=8, ha='right', va='bottom')

    plt.savefig(Save_loc + '/concentration_yield_labels.png')


def plot_concentration_spacing(x, y,Save_loc,directory_name):
    x = pd.to_numeric(pd.Series(x), errors='coerce')
    y = pd.to_numeric(pd.Series(y), errors='coerce')
    mask = x.notna() & y.notna()
    plt.figure(figsize=(15, 10))
    plt.scatter(x[mask], y[mask])
    plt.xlabel("Concentration ")
    plt.ylabel("Spacing")
    plt.title("Concentration Vs spacing " + directory_name)

    #plt.xscale('log')
    plt.xticks([0.001, 0.005, 0.05, 1, 0.07, 0.2, 2, 4, 0.4, 0.1],
               ["0.001", "0.005", "0.05", " 1", "0.07", "0.2", "2", "4", "0.4", "0.1"])
    plt.savefig(Save_loc + '/concentration_spacing.png')


def Spacing_yield_labels(x, y, labels,Save_loc,directory_name):
    plt.figure(figsize=(15, 10))
    x = pd.to_numeric(pd.Series(x), errors='coerce')
    y = pd.to_numeric(pd.Series(y), errors='coerce')
    labels = list(labels)
    mask = (x.notna() & y.notna() & (y != 0)).values
    x_f = x[mask]
    y_f = y[mask]
    labels_f = [labels[i] for i, m in enumerate(mask) if m]
    plt.scatter(x_f, y_f)
    for i in range(len(x_f)):
        plt.text(x_f.iloc[i], y_f.iloc[i], labels_f[i], fontsize=8, ha='right', va='bottom')



    plt.xlabel("Spacing ")
    plt.ylabel("Yield")
    plt.title("spacing Vs yield " + directory_name)

    # plt.xscale('log')
    # plt.xticks([0.001,0.005, 0.05, 1, 0.07,0.2, 2,4,0.4,0.1],["0.001","0.005", "0.05"," 1", "0.07","0.2", "2","4","0.4","0.1"])
    plt.savefig(Save_loc + '/Spacing_yield_labels.png')


def Spacing_yield(x, y,Save_loc,directory_name):
    plt.figure(figsize=(15, 10))
    x = pd.to_numeric(pd.Series(x), errors='coerce')
    y = pd.to_numeric(pd.Series(y), errors='coerce')
    mask = x.notna() & y.notna()
    plt.scatter(x[mask], y[mask])

    plt.xlabel("Spacing ")
    plt.ylabel("Yield")
    plt.title("spacing Vs yield " + directory_name)

    # plt.xscale('log')
    # plt.xticks([0.001,0.005, 0.05, 1, 0.07,0.2, 2,4,0.4,0.1],["0.001","0.005", "0.05"," 1", "0.07","0.2", "2","4","0.4","0.1"])
    plt.savefig(Save_loc + '/Spacing_yield.png')


def facet_concentration_yield_by_polymer(df: pd.DataFrame, save_loc: str, directory_name: str) -> None:
    # Ensure numeric
    df = df.copy()
    df['Np Concentration'] = pd.to_numeric(df['Np Concentration'], errors='coerce')
    df['Yield'] = pd.to_numeric(df['Yield'], errors='coerce')
    df = df.dropna(subset=['Np Concentration', 'Yield'])
    if 'Polymer' not in df.columns:
        return
    g = sns.relplot(
        data=df,
        x='Np Concentration', y='Yield', col='Polymer', kind='scatter', col_wrap=3, height=4, facet_kws={'sharex': False, 'sharey': True}
    )
    g.set_titles("{col_name}")
    g.fig.suptitle('Concentration vs Yield by Polymer ' + directory_name, y=1.03)
    g.savefig(save_loc + '/facet_concentration_yield_by_polymer.png')


def facet_spacing_yield_by_polymer(df: pd.DataFrame, save_loc: str, directory_name: str) -> None:
    df = df.copy()
    if 'Polymer' not in df.columns:
        return
    df['Qd Spacing (nm)'] = pd.to_numeric(df['Qd Spacing (nm)'], errors='coerce')
    df['Yield'] = pd.to_numeric(df['Yield'], errors='coerce')
    df = df.dropna(subset=['Qd Spacing (nm)', 'Yield'])
    g = sns.relplot(
        data=df,
        x='Qd Spacing (nm)', y='Yield', col='Polymer', kind='scatter', col_wrap=3, height=4, facet_kws={'sharex': False, 'sharey': True}
    )
    g.set_titles("{col_name}")
    g.fig.suptitle('Spacing vs Yield by Polymer ' + directory_name, y=1.03)
    g.savefig(save_loc + '/facet_spacing_yield_by_polymer.png')


def correlation_heatmap(df: pd.DataFrame, save_loc: str) -> None:
    numeric_cols = ['Yield', 'Np Concentration', 'Qd Spacing (nm)', 'Volume Fraction', 'Volume Fraction %', 'Weight Fraction']
    avail = [c for c in numeric_cols if c in df.columns]
    if not avail:
        return
    data = df[avail].apply(pd.to_numeric, errors='coerce')
    corr = data.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc, 'correlation_heatmap.png'))
    plt.close()


def pairplot_numeric(df: pd.DataFrame, save_loc: str) -> None:
    numeric_cols = ['Yield', 'Np Concentration', 'Qd Spacing (nm)', 'Volume Fraction', 'Volume Fraction %', 'Weight Fraction']
    avail = [c for c in numeric_cols if c in df.columns]
    if len(avail) < 2:
        return
    data = df[avail].apply(pd.to_numeric, errors='coerce').dropna()
    g = sns.pairplot(data, diag_kind='kde')
    g.fig.suptitle('Pairplot of Numeric Variables', y=1.02)
    g.savefig(os.path.join(save_loc, 'pairplot_numeric.png'))
    plt.close()


def violin_yield_by_polymer(df: pd.DataFrame, save_loc: str) -> None:
    if 'Polymer' not in df.columns:
        return
    data = df[['Polymer', 'Yield']].copy()
    data['Yield'] = pd.to_numeric(data['Yield'], errors='coerce')
    data = data.dropna()
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, x='Polymer', y='Yield')
    plt.title('Yield distribution by Polymer')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc, 'violin_yield_by_polymer.png'))
    plt.close()


def box_yield_by_polymer(df: pd.DataFrame, save_loc: str) -> None:
    if 'Polymer' not in df.columns:
        return
    data = df[['Polymer', 'Yield']].copy()
    data['Yield'] = pd.to_numeric(data['Yield'], errors='coerce')
    data = data.dropna()
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='Polymer', y='Yield')
    plt.title('Yield boxplot by Polymer')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc, 'box_yield_by_polymer.png'))
    plt.close()

