import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


file_path = "distribute_central/result.csv"
data = pd.read_csv(file_path)


fidelity_values = data["key_per_time"].values


num_parts_x = 20
num_parts_y = 20


d_As = np.linspace(2, 130, num=num_parts_x)
T_2s = np.linspace(0.001, 1, num=num_parts_y)

fidelity_grid = fidelity_values.reshape((num_parts_x, num_parts_y)).T


plt.figure(figsize=(8, 6))
plt.pcolormesh(
    d_As, T_2s, fidelity_grid, shading="auto", norm=mcolors.CenteredNorm(), cmap="bwr"
)
plt.colorbar(label="rate_per_time")
plt.ylabel("T_2")
plt.xlabel("d_A")
# plt.title('fidelities')
plt.xlim(d_As.min(), d_As.max())
plt.ylim(T_2s.min(), T_2s.max())
xticks = np.append(d_As[::4], d_As[-1])
yticks = np.append(T_2s[::4], T_2s[-1])
plt.xticks(xticks)
plt.yticks(yticks)
plt.show()
