import matplotlib.pyplot as plt

def plot_graph(x_data,y_data,xlabel = "x",ylabel = "y" ,label="Undefined",xscale="",yscale="",yerr= None):
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

plot_graph(1,2,"x","y",yscale='log')