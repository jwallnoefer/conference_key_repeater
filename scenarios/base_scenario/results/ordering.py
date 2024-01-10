import re
import matplotlib.pyplot as plt
import numpy as np


def kilo(list1):
    list2 = []
    for x in list1:
        list2.append(x / 1000)

    return list2


# define a function to extract lengths and keyrates
def extract_lengths_and_keyrates(line):
    match = re.search(r"Task \d+: Length = ([\d.]+), Rate = ([\d.-]+)", line)
    if match:
        length = float(match.group(1))
        keyrate = float(match.group(2))
        return length, keyrate
    return None


# read the original text document
# with open("num_results.txt", "r") as file:
#     lines = file.readlines()

# lists to store the extracted data
lengths = []
keyrates = []

# iterate through the lines and extract data
# for line in lines:
#     info = extract_lengths_and_keyrates(line)
#     if info:
#         length, keyrate = info
#         lengths.append(length)
#         keyrates.append(keyrate)


## load multipartite analytic results with memories
ana_results = np.load(
    "/Users/janki/requsim/conference_key_repeater/scenarios/base_scenario/results/ana_results.npz"
)
ana_lengths = ana_results["array1"]
ana_key_rates = ana_results["array2"]

# plot everything
plt.plot(
    ana_lengths,
    kilo(ana_key_rates),
    "o",
    color="red",
    ms=1,
    label="analytic, multipartite with mem",
)
# plt.scatter(kilo(lengths_num_new), kilo(rates_num_new), s=1, marker='o', color='darkseagreen', label='numeric, multipartite with mem')
plt.xlabel("distance to central station [km]")
plt.ylabel("key rate [Kbits/s]")
plt.yscale("log")
plt.ylim(10 ** (-5), 10**3)
plt.legend()
plt.show()
