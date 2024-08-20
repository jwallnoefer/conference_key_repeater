# # fidelity plot #####
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from matplotlib.colors import LogNorm
# from matplotlib.ticker import ScalarFormatter

# # Read data from CSV
# file_path = "P_unis_T2_m_nocut/distribute_central/result.csv"
# data = pd.read_csv(file_path)

# gesucht = "fidelity"
# fidelity_values = data[gesucht].values

# # Define grid parameters
# num_parts_x = 20
# num_parts_y = 20
# # d_As = np.linspace(2e3, 130e3, num=num_parts_x)
# # T_2s = np.logspace(-3, 1, num=num_parts_y)

# ms = np.linspace(1, 20, num=num_parts_x)  # x-axis
# T_2s = np.logspace(-3, 1, num=num_parts_y)  # y-axis

# param_matrix = np.array(np.meshgrid(ms, T_2s)).T.reshape(-1, 2)

# # Create parameter grid and reshape fidelity values
# #param_matrix = np.array(np.meshgrid(d_As, T_2s)).T.reshape(-1, 2)
# param_matrix = np.array(np.meshgrid(ms, T_2s)).T.reshape(-1, 2)
# fidelity_grid = fidelity_values.reshape((num_parts_x, num_parts_y)).T

# # Plotting
# plt.figure(figsize=(8, 6))
# pcm = plt.pcolormesh(ms, T_2s, fidelity_grid, cmap="viridis", vmin=0.48, vmax=1)


# ticks = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1])
# cbar = plt.colorbar(pcm, pad=0.04, shrink=0.8, aspect=10)
# # # Apply the ticks to the colorbar
# cbar.set_ticks(ticks)
# cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
# cbar.set_label('Fidelity')
# plt.ylabel("T_2 (s)")
# #plt.yscale("log")
# plt.xlabel("d_A (km)")
# plt.title("Fidelity no cut")
# # plt.xlim(d_As.min(), d_As.max())
# # plt.ylim(T_2s.min(), T_2s.max())
# # # Set x and y ticks
# # xticks_m = [2e3, 25e3, 50e3, 75e3, 100e3, 125e3]
# # xtick_labels_km = [f'{int(tick/1e3)}' for tick in xticks_m]
# # plt.xticks(xticks_m, labels=xtick_labels_km)
# plt.xlabel("d_A (km)")

# # yticks = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
# # plt.yticks(yticks, labels=[r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])

# plt.show()

### key per time plot #####

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from matplotlib.colors import SymLogNorm

# # Read data from CSV
# file_path = "P_unis_T2_m_nocut/distribute_central/result.csv"
# data = pd.read_csv(file_path)

# gesucht = "key_per_time"
# fidelity_values = data[gesucht].values

# num_parts_x = 20
# num_parts_y = 20
# # d_As = np.linspace(2e3, 130e3, num=num_parts_x)
# # T_2s = np.logspace(-3, 1, num=num_parts_y)

# # num_parts_x = 6
# # num_parts_y = 5
# # num_parts = num_parts_x*num_parts_y

# # Ns = np.linspace(4, 9, num=num_parts_x)  # x-axis
# # ms = np.linspace(4, 20, num=num_parts_y)  # y-axis

# ms = np.linspace(1, 20, num=num_parts_x)  # x-axis
# T_2s = np.logspace(-3, 1, num=num_parts_y)  # y-axis


# param_matrix = np.array(np.meshgrid(ms, T_2s)).T.reshape(-1, 2)
# fidelity_grid = fidelity_values.reshape((num_parts_x, num_parts_y)).T

# # normalization
# linthresh = 0.03  # Linear threshold
# linscale = 0.03   # Linear scale
# vmin = np.min(fidelity_grid)
# vmax = np.max(fidelity_grid)

# norm = SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=1e3, base=10)

# # create colormap
# cmap = plt.get_cmap('Spectral')

# # Plotting
# plt.figure(figsize=(8, 6))

# # use SymLogNorm normalization
# pcm = plt.pcolormesh(ms, T_2s, fidelity_grid, cmap=cmap, norm=norm)

# # create and format colorbar
# cbar = plt.colorbar(pcm, label=gesucht)
# ticks = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e2, 1e3])
# cbar.set_ticks(ticks)

# # Manually set the ticks on the colorbar
# # ticks = np.concatenate((
# #     -np.logspace(np.log10(linthresh), np.log10(vmax), num=5),
# #     [0],
# #     np.logspace(np.log10(linthresh), np.log10(vmax), num=5)
# # ))
# # cbar.set_ticks(ticks)
# # cbar.set_ticklabels([f'{tick:.1e}' for tick in ticks])

# plt.ylabel("T_2")
# plt.yscale("log")
# plt.xlabel("m")
# plt.title("key per time, sym 70 no cutoff")
# # plt.xlim(d_As.min(), d_As.max())
# # plt.ylim(T_2s.min(), T_2s.max())
# #plt.yscale("log")

# #Set x and y ticks
# xticks = [5, 10, 15, 20]
# plt.xticks(xticks)
# # plt.yticks(yticks)

# plt.show()

### modified key per time plot, take out all negative values ####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import SymLogNorm

# Read data from CSV
file_path = "P_unis_T2_m_nocut/distribute_central/result.csv"
data = pd.read_csv(file_path)

gesucht = "key_per_time"
fidelity_values = data[gesucht].values

num_parts_x = 20
num_parts_y = 20
# d_As = np.linspace(2e3, 130e3, num=num_parts_x)
# T_2s = np.logspace(-3, 1, num=num_parts_y)

# num_parts_x = 6
# num_parts_y = 5
# num_parts = num_parts_x*num_parts_y

# Ns = np.linspace(4, 9, num=num_parts_x)  # x-axis
# ms = np.linspace(4, 20, num=num_parts_y)  # y-axis

ms = np.linspace(1, 20, num=num_parts_x)  # x-axis
T_2s = np.logspace(-3, 1, num=num_parts_y)  # y-axis


param_matrix = np.array(np.meshgrid(ms, T_2s)).T.reshape(-1, 2)
fidelity_grid = fidelity_values.reshape((num_parts_x, num_parts_y)).T
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from matplotlib.colors import SymLogNorm

# # Read data from CSV
# file_path = "P_asym_dA_logT2_cut0.05/distribute_central/result.csv"
# data = pd.read_csv(file_path)

# gesucht = "key_per_time"
# fidelity_values = data[gesucht].values

# num_parts_x = 20
# num_parts_y = 20
# d_As = np.linspace(2e3, 130e3, num=num_parts_x)
# T_2s = np.logspace(-3, 1, num=num_parts_y)

# param_matrix = np.array(np.meshgrid(d_As, T_2s)).T.reshape(-1, 2)
# fidelity_grid = fidelity_values.reshape((num_parts_x, num_parts_y)).T

# Replace negative values with np.nan
fidelity_grid[fidelity_grid < 0] = np.nan

# Define colormap with a specific color for NaNs (e.g., white)
base_cmap = plt.get_cmap("Spectral")
colors = base_cmap(np.linspace(0, 1, 256))

# Add white color for NaNs
nan_color = np.array([1, 1, 1, 1])  # RGBA for white
new_colors = np.vstack((colors, nan_color))
custom_cmap = mcolors.ListedColormap(new_colors)

# Define normalization (logarithmic scale)
linthresh = 0.03
linscale = 0.03
vmin = np.nanmin(fidelity_grid[np.isfinite(fidelity_grid)])
vmax = np.nanmax(fidelity_grid[np.isfinite(fidelity_grid)])

norm = SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=vmax, base=10)

# Plotting
plt.figure(figsize=(8, 6))

# Use custom colormap and normalization
pcm = plt.pcolormesh(
    ms, T_2s, fidelity_grid, cmap=custom_cmap, norm=norm, shading="auto"
)

# Create and format colorbar
cbar = plt.colorbar(pcm, pad=0.04, shrink=0.8, aspect=10)

# Define custom ticks for the colorbar
ticks = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])

# Apply the ticks to the colorbar
cbar.set_ticks(ticks)

# Set tick labels with proper formatting
tick_labels = [f"{tick:.0e}" for tick in ticks]
cbar.set_ticklabels(tick_labels)

# Set label for colorbar
cbar.set_label(gesucht)

# Adjust the appearance
plt.ylabel("T_2")
plt.xlabel("d_A")
plt.title("Key per Time")
plt.xlim(ms.min(), ms.max())
plt.ylim(T_2s.min(), T_2s.max())
plt.yscale("log")

# Set x and y ticks
# xticks_m = [2e3, 25e3, 50e3, 75e3, 100e3, 125e3]
# xtick_labels_km = [f'{int(tick/1e3)}' for tick in xticks_m]
# plt.xticks(xticks_m, labels=xtick_labels_km)
# plt.xlabel("d_A (km)")

yticks = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
plt.yticks(
    yticks, labels=[r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"]
)

plt.show()


### modified, replace all negative values with 0
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from matplotlib.colors import SymLogNorm, BoundaryNorm
# from matplotlib.colorbar import ColorbarBase

# # Read data from CSV
# file_path = "P_asym_dA_logT2_cut0.05/distribute_central/result.csv"
# data = pd.read_csv(file_path)

# gesucht = "key_per_time"
# fidelity_values = data[gesucht].values

# num_parts_x = 20
# num_parts_y = 20
# d_As = np.linspace(2e3, 130e3, num=num_parts_x)
# T_2s = np.logspace(-3, 1, num=num_parts_y)

# param_matrix = np.array(np.meshgrid(d_As, T_2s)).T.reshape(-1, 2)
# fidelity_grid = fidelity_values.reshape((num_parts_x, num_parts_y)).T

# # Set all negative values to 0
# fidelity_grid[fidelity_grid < 0] = 0

# # Define colormap
# base_cmap = plt.get_cmap('Spectral')
# colors = base_cmap(np.linspace(0, 1, 256))

# # Add white color for zero
# new_colors = np.vstack(([[1, 1, 1, 1]], colors))
# custom_cmap = mcolors.ListedColormap(new_colors)

# # Custom normalization
# linthresh = 0.03
# linscale = 0.03
# vmin = np.min(fidelity_grid)
# vmax = np.max(fidelity_grid)

# norm = SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=vmax, base=10)

# plt.figure(figsize=(8, 6))

# # Use custom colormap and normalization
# pcm = plt.pcolormesh(d_As, T_2s, fidelity_grid, cmap=custom_cmap, norm=norm, shading='auto')

# # Create and format colorbar
# cbar = plt.colorbar(pcm, pad=0.04, shrink=0.8, aspect=10)

# # Define custom ticks for the colorbar
# ticks = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])

# # Apply the ticks to the colorbar
# cbar.set_ticks(ticks)

# # Set tick labels with scientific notation
# tick_labels = [f'$10^{{{int(np.log10(tick))}}}$' for tick in ticks]

# cbar.set_label(gesucht)

# # Adjust the appearance
# plt.ylabel("T_2 (s)")
# plt.title("Key per Time (bits/s) cut = 0.05s")
# plt.xlim(d_As.min(), d_As.max())
# plt.ylim(T_2s.min(), T_2s.max())
# plt.yscale("log")

# # Set x and y ticks
# xticks_m = [2e3, 25e3, 50e3, 75e3, 100e3, 125e3]
# xtick_labels_km = [f'{int(tick/1e3)}' for tick in xticks_m]
# plt.xticks(xticks_m, labels=xtick_labels_km)
# plt.xlabel("d_A (km)")

# yticks = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
# plt.yticks(yticks, labels=[r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])

# plt.show()
