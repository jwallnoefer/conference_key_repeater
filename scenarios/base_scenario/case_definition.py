from base_scenario import run
import numpy as np


def run_func(case=None, part=None):
    return run


scenario_name = "base_scenario"
cases = []

# # Template
# case_specification = {
#     "name": str,
#     "subcase_name": str,
#     "num_parts": int,
#     "index": array_like,
#     "case_args": {part: {"distance_from_central": lengths[part],
#                          "num_parties": 4
#                          "max_iter": 1e5,
#                          "params": {"P_LINK": 0.01,
#                                       "F_INIT": 0.99,
#                                       "T_P": 1e-6,
#                                       "P_D": 1e-6,
#                                       "T_DP": 100e-3,
#                                       },
#                          }
#                   for part in range(num_parts)
#                   }
# }

case_name = "example_case"
num_parts = 128
max_iter = 100  # 1e5
num_parties = 4
base_params = {
    "P_LINK": 0.01,
    "F_INIT": 0.99,
    "T_P": 1e-6,  # preparation time, consistent with MHz entangled pair source
    "P_D": 1e-6,  # dark count probability
    "T_DP": 100e-3,  # dephasing time
}
lengths = np.linspace(10e3, 200e3, num=num_parts)

case_specification = {
    "name": case_name,
    "subcase_name": "example_subcase",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {
        part: {
            "distance_from_central": lengths[part],
            "num_parties": 4,
            "max_iter": 1e5,
            "params": {
                "P_LINK": 0.01,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 100e-3,
            },
        }
        for part in range(num_parts)
    },
}


def case_args(case, part):
    return cases[case]["case_args"][part]


def name(case):
    return cases[case]["name"]


def num_parts(case):
    return cases[case]["num_parts"]


def subcase_name(case):
    return cases[case]["subcase_name"]


def index(case):
    return cases[case]["index"]
