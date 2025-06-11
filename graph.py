import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


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
    #fig.show()
    #plt.pause(0.1)

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

    plt.figure(figsize=(15, 10))
    plt.scatter(x, y)
    plt.xlabel("Concentration ")
    plt.ylabel("Yield")
    plt.title("Concentration Vs Yield " + directory_name)

    #plt.xscale('log')
    plt.xticks([0.001, 0.005, 0.05, 1, 0.07, 0.2, 2, 4, 0.4, 0.1],
               ["0.001", "0.005", "0.05", " 1", "0.07", "0.2", "2", "4", "0.4", "0.1"])

    plt.savefig(Save_loc + '/concentration_yield.png')



def plot_concentration_yield_labels(x, y, labels,Save_loc,directory_name):
    plt.figure(figsize=(15, 10))
    plt.scatter(x, y)
    plt.xlabel("Concentration ")
    plt.ylabel("Yield")
    plt.title("Concentration Vs Yield " + directory_name )

    #plt.xscale('log')
    plt.xticks([0.001, 0.005, 0.05, 1, 0.07, 0.2, 2, 4, 0.4, 0.1, 0.01],
               ["0.001", "0.005", "0.05", " 1", "0.07", "0.2", "2", "4", "0.4", "0.1", "0.01"])

    # Extract the first part of each label before the "-"
    labels_filtered = [label.split('-')[0] for label in labels]

    # Add labels to each point
    for i in range(len(x)):
        plt.text(x[i], y[i], labels_filtered[i], fontsize=8, ha='right', va='bottom')

    plt.savefig(Save_loc + '/concentration_yield_labels.png')


def plot_concentration_spacing(x, y,Save_loc,directory_name):
    plt.figure(figsize=(15, 10))
    plt.scatter(x, y)
    plt.xlabel("Concentration ")
    plt.ylabel("Spacing")
    plt.title("Concentration Vs spacing " + directory_name)

    #plt.xscale('log')
    plt.xticks([0.001, 0.005, 0.05, 1, 0.07, 0.2, 2, 4, 0.4, 0.1],
               ["0.001", "0.005", "0.05", " 1", "0.07", "0.2", "2", "4", "0.4", "0.1"])
    plt.savefig(Save_loc + '/concentration_yield.png')


def Spacing_yield_labels(x, y, labels,Save_loc,directory_name):

    plt.figure(figsize=(15, 10))

    # Convert to NumPy arrays (for easier filtering)
    x = np.array(x)
    y = np.array(y)
    labels = np.array(labels)

    # Remove zero values from y (and corresponding x values)
    mask = y != 0  # Boolean mask where y ≠ 0
    x_filtered = x[mask]
    y_filtered = y[mask]
    labels_filtered = labels[mask]  # Keep only labels for valid points

    plt.scatter(x_filtered, y_filtered)

    # Label each point with the device name
    for i in range(len(x_filtered)):
        plt.text(x_filtered[i], y_filtered[i], labels_filtered[i], fontsize=8, ha='right', va='bottom')



    plt.xlabel("Spacing ")
    plt.ylabel("Yield")
    plt.title("spacing Vs yield " + directory_name)

    # plt.xscale('log')
    # plt.xticks([0.001,0.005, 0.05, 1, 0.07,0.2, 2,4,0.4,0.1],["0.001","0.005", "0.05"," 1", "0.07","0.2", "2","4","0.4","0.1"])
    plt.savefig(Save_loc + '/Spacing_yield_labels.png')


def Spacing_yield(x, y,Save_loc,directory_name):
    plt.figure(figsize=(15, 10))
    # Convert to NumPy arrays (for easier filtering)
    # x = np.array(x)
    # y = np.array(y)
    #
    # # Remove zero values from y (and corresponding x values)
    # mask = y != 0  # Boolean mask where y ≠ 0
    # x_filtered = x[mask]
    # y_filtered = y[mask]

    plt.scatter(x, y)

    plt.xlabel("Spacing ")
    plt.ylabel("Yield")
    plt.title("spacing Vs yield " + directory_name)

    # plt.xscale('log')
    # plt.xticks([0.001,0.005, 0.05, 1, 0.07,0.2, 2,4,0.4,0.1],["0.001","0.005", "0.05"," 1", "0.07","0.2", "2","4","0.4","0.1"])
    plt.savefig(Save_loc + '/Spacing_yield.png')

