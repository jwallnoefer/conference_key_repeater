import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define the base directories and subdirectories
base_dirs = {
    "P_sym150_per_TCUT_1mem_v4": {"marker": "o", "size": 5},
    # 'P_sym_cut0.15_5mem': {'marker': '^', 'size': 15}
}

sub_dirs = {
    "distribute_central": "blue",
    "distribute_outer": "orange",
    "measure_central": "green",
    "measure_outer": "red",
}

# Loop through each base directory
for base_dir, base_info in base_dirs.items():
    marker = base_info["marker"]
    size = base_info["size"]

    # Loop through each subdir
    for sub_dir, color in sub_dirs.items():

        file_path = os.path.join(base_dir, sub_dir, "result.csv")

        if os.path.exists(file_path):
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path, index_col=0)

            x = df.index
            y = df["fidelity"]

            # Plot data
            # plt.scatter(x/1000, y, s=size, color=color, label=f'{base_dir} - {sub_dir}', marker=marker)
            plt.scatter(x, y, s=size, color=color, label=f"{sub_dir}", marker=marker)

# legend_handles = [
#     Line2D([0], [0], color=color, lw=1, label=f'{sub_dir}') for sub_dir, color in sub_dirs.items()
# ] + [
#     Line2D([0], [0], marker=base_dirs['P_sym_cut0.15_5mem']['marker'], color='black', label='TCUT=0.15s', markerfacecolor='black', markersize=7, linestyle='None')
# ]

# Add custom legend
# plt.legend(handles=legend_handles, loc='lower left', ncol=1, frameon=True)

plt.legend()
# Add labels and title
plt.xlabel("T_CUT (s)")
## key per time
plt.ylabel("Key per time (bits/s)")
# plt.yscale('log')
## fidelity
# plt.ylabel('Fidelity')

plt.title("Symmetric 150km, 1mem")
plt.grid(True)

plt.show()
