import re
import matplotlib.pyplot as plt
import numpy as np


def kilo(list1):
    list2 = []
    for x in list1:
        list2.append(x / 1000)

    return list2


#  extract lengths and keyrates from output file from cluster
def extract_lengths_and_keyrates(line):
    match = re.search(r"Task \d+: Length = ([\d.]+), Rate = ([\d.-]+)", line)
    if match:
        length = float(match.group(1))
        keyrate = float(match.group(2))
        return length, keyrate
    return None


with open("num_keyrates.rtf", "r") as file:
    lines = file.readlines()

lengths = []
keyrates = []

for line in lines:
    info = extract_lengths_and_keyrates(line)
    if info:
        length, keyrate = info
        lengths.append(length)
        keyrates.append(keyrate)


## load multipartite analytic results with memories
ana_results = np.load(
    "/Users/janki/requsim/conference_key_repeater/scenarios/base_scenario/results/ana_results_original_protocol.npz"
)
ana_lengths = ana_results["array1"]
ana_key_rates = ana_results["array2"]


# load multipartite analytic results with memories
ana_results2 = np.load(
    "/Users/janki/requsim/conference_key_repeater/scenarios/base_scenario/results/ana_results_bobs_measure_directly.npz"
)
ana_lengths2 = ana_results2["array1"]
ana_key_rates2 = ana_results2["array2"]


# plot everything
plt.plot(
    ana_lengths,
    kilo(ana_key_rates),
    "o",
    color="red",
    ms=1,
    label="analytic, original protocol",
)

plt.plot(
    ana_lengths2,
    kilo(ana_key_rates2),
    "o",
    color="orange",
    ms=1,
    label="analytic, Bobs measure directly",
)

plt.scatter(
    kilo(lengths),
    kilo(keyrates),
    s=1,
    marker="o",
    color="indigo",
    label="numeric, mixed CC",
)
plt.xlabel("distance to central station [km]")
plt.ylabel("key rate [Kbits/s]")
plt.yscale("log")
plt.ylim(10 ** (-5), 10**3)
plt.legend()
plt.savefig("Keyrates_comparison.png", dpi=200)
plt.show()
