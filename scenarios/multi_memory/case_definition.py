from multi_memory import run
import numpy as np


def run_func(case=None, part=None):
    return run


scenario_name = "multi_memory"
cases = []

case_name = "P_sym_nocut_1mem"
num_parts = 64
max_iter = 10
num_parties = 4
base_params = {
    "P_LINK": 1,
    "F_INIT": 0.99,
    "T_P": 1e-6,  # preparation time, consistent with MHz entangled pair source
    "P_D": 1e-6,  # dark count probability
    "T_DP": 1,  # dephasing time
    "T_CUT": None,
}
num_memories = 1
# mode = "distribute"
# source_position = "outer"

lengths = np.linspace(1e3, 220e3, num=num_parts)

case_specification_1 = {
    "name": case_name,
    "subcase_name": "distribute_central",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {
        part: {
            "distance_from_central": [lengths[part]] * int(num_parties - 1),
            "distance_A": lengths[part],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": None,
            },
            "num_memories": num_memories,
            "mode": "distribute",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_1)


case_specification_2 = {
    "name": case_name,
    "subcase_name": "distribute_outer",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {
        part: {
            "distance_from_central": [lengths[part]] * int(num_parties - 1),
            "distance_A": lengths[part],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": None,
            },
            "num_memories": num_memories,
            "mode": "distribute",
            "source_position": "outer",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_2)


case_specification_3 = {
    "name": case_name,
    "subcase_name": "measure_central",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {
        part: {
            "distance_from_central": [lengths[part]] * int(num_parties - 1),
            "distance_A": lengths[part],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": None,
            },
            "num_memories": num_memories,
            "mode": "measure",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_3)


case_specification_4 = {
    "name": case_name,
    "subcase_name": "measure_outer",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {
        part: {
            "distance_from_central": [lengths[part]] * int(num_parties - 1),
            "distance_A": lengths[part],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": None,
            },
            "num_memories": num_memories,
            "mode": "measure",
            "source_position": "outer",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_4)


case_name = "P_sym_cut0.3_1mem"
num_parts = 64
max_iter = 10**5
num_parties = 4
base_params = {
    "P_LINK": 1,
    "F_INIT": 0.99,
    "T_P": 1e-6,  # preparation time, consistent with MHz entangled pair source
    "P_D": 1e-6,  # dark count probability
    "T_DP": 1,  # dephasing time
    "T_CUT": 0.3,  # cut off time
}
num_memories = 1
# mode = "distribute"
# source_position = "outer"


lengths = np.linspace(1e3, 220e3, num=num_parts)

case_specification_1 = {
    "name": case_name,
    "subcase_name": "distribute_central",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {
        part: {
            "distance_from_central": [lengths[part]] * int(num_parties - 1),
            "distance_A": lengths[part],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": 0.3,
            },
            "num_memories": num_memories,
            "mode": "distribute",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_1)


case_specification_2 = {
    "name": case_name,
    "subcase_name": "distribute_outer",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {
        part: {
            "distance_from_central": [lengths[part]] * int(num_parties - 1),
            "distance_A": lengths[part],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": 0.3,
            },
            "num_memories": num_memories,
            "mode": "distribute",
            "source_position": "outer",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_2)


case_specification_3 = {
    "name": case_name,
    "subcase_name": "measure_central",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {
        part: {
            "distance_from_central": [lengths[part]] * int(num_parties - 1),
            "distance_A": lengths[part],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": 0.3,
            },
            "num_memories": num_memories,
            "mode": "measure",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_3)


case_specification_4 = {
    "name": case_name,
    "subcase_name": "measure_outer",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {
        part: {
            "distance_from_central": [lengths[part]] * int(num_parties - 1),
            "distance_A": lengths[part],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": 0.3,
            },
            "num_memories": num_memories,
            "mode": "measure",
            "source_position": "outer",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_4)

case_name = "P_asym_dA_logT2_nocut"
num_parts_x = 20
num_parts_y = 20
num_parts = num_parts_x * num_parts_y
max_iter = 10**5
num_parties = 4
d_B = 4e3
num_memories = 1
T_CUT = None


d_As = np.linspace(2e3, 130e3, num=num_parts_x)  # x-axis
T_2s = np.logspace(-3, 1, num=num_parts_y)  # y-axis

param_matrix = np.array(np.meshgrid(d_As, T_2s)).T.reshape(-1, 2)

case_specification_1 = {
    "name": case_name,
    "subcase_name": "distribute_central",
    "num_parts": num_parts,
    "index": list(range(num_parts)),
    "case_args": {
        part: {
            "distance_from_central": [d_B] * int(num_parties - 1),
            "distance_A": param_matrix[part][0],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": param_matrix[part][1],
                "T_CUT": T_CUT,
            },
            "num_memories": num_memories,
            "mode": "distribute",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}

cases.append(case_specification_1)

case_name = "P_asym_dA_logT2_cut0.2"
num_parts_x = 20
num_parts_y = 20
num_parts = num_parts_x * num_parts_y
max_iter = 10**5
num_parties = 4
d_B = 4e3
num_memories = 1
T_CUT = 0.2


d_As = np.linspace(2e3, 130e3, num=num_parts_x)  # x-axis
T_2s = np.logspace(-3, 1, num=num_parts_y)  # y-axis

param_matrix = np.array(np.meshgrid(d_As, T_2s)).T.reshape(-1, 2)

case_specification_1 = {
    "name": case_name,
    "subcase_name": "distribute_central",
    "num_parts": num_parts,
    "index": list(range(num_parts)),
    "case_args": {
        part: {
            "distance_from_central": [d_B] * int(num_parties - 1),
            "distance_A": param_matrix[part][0],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": param_matrix[part][1],
                "T_CUT": T_CUT,
            },
            "num_memories": num_memories,
            "mode": "distribute",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}


cases.append(case_specification_1)


case_name = "P_sym_cut0.1_5mem"
num_parts = 64
max_iter = 10**5
num_parties = 4
base_params = {
    "P_LINK": 1,
    "F_INIT": 0.99,
    "T_P": 1e-6,  # preparation time, consistent with MHz entangled pair source
    "P_D": 1e-6,  # dark count probability
    "T_DP": 1,  # dephasing time
}
num_memories = 5
T_CUT = 0.1
# mode = "distribute"
# source_position = "outer"


lengths = np.linspace(1e3, 220e3, num=num_parts)

case_specification_1 = {
    "name": case_name,
    "subcase_name": "distribute_central",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {
        part: {
            "distance_from_central": [lengths[part]] * int(num_parties - 1),
            "distance_A": lengths[part],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUT,
            },
            "num_memories": num_memories,
            "mode": "distribute",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_1)


case_specification_2 = {
    "name": case_name,
    "subcase_name": "distribute_outer",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {
        part: {
            "distance_from_central": [lengths[part]] * int(num_parties - 1),
            "distance_A": lengths[part],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUT,
            },
            "num_memories": num_memories,
            "mode": "distribute",
            "source_position": "outer",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_2)


case_specification_3 = {
    "name": case_name,
    "subcase_name": "measure_central",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {
        part: {
            "distance_from_central": [lengths[part]] * int(num_parties - 1),
            "distance_A": lengths[part],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUT,
            },
            "num_memories": num_memories,
            "mode": "measure",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_3)


case_specification_4 = {
    "name": case_name,
    "subcase_name": "measure_outer",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {
        part: {
            "distance_from_central": [lengths[part]] * int(num_parties - 1),
            "distance_A": lengths[part],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUT,
            },
            "num_memories": num_memories,
            "mode": "measure",
            "source_position": "outer",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_4)


case_name = "P_sym150_per_TCUT_5mem"
num_parts = 64
max_iter = 10**5
num_parties = 4
length = 150e3
base_params = {
    "P_LINK": 1,
    "F_INIT": 0.99,
    "T_P": 1e-6,  # preparation time, consistent with MHz entangled pair source
    "P_D": 1e-6,  # dark count probability
    "T_DP": 1,  # dephasing time
}
num_memories = 5


T_CUTS = np.linspace(0.005, 0.3, num=num_parts)

case_specification_1 = {
    "name": case_name,
    "subcase_name": "distribute_central",
    "num_parts": num_parts,
    "index": T_CUTS,
    "case_args": {
        part: {
            "distance_from_central": [length] * int(num_parties - 1),
            "distance_A": length,
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUTS[part],
            },
            "num_memories": num_memories,
            "mode": "distribute",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_1)

case_specification_2 = {
    "name": case_name,
    "subcase_name": "distribute_outer",
    "num_parts": num_parts,
    "index": T_CUTS,
    "case_args": {
        part: {
            "distance_from_central": [length] * int(num_parties - 1),
            "distance_A": length,
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUTS[part],
            },
            "num_memories": num_memories,
            "mode": "distribute",
            "source_position": "outer",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_2)


case_specification_3 = {
    "name": case_name,
    "subcase_name": "measure_central",
    "num_parts": num_parts,
    "index": T_CUTS,
    "case_args": {
        part: {
            "distance_from_central": [length] * int(num_parties - 1),
            "distance_A": length,
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUTS[part],
            },
            "num_memories": num_memories,
            "mode": "measure",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_3)


case_specification_4 = {
    "name": case_name,
    "subcase_name": "measure_outer",
    "num_parts": num_parts,
    "index": T_CUTS,
    "case_args": {
        part: {
            "distance_from_central": [length] * int(num_parties - 1),
            "distance_A": length,
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUTS[part],
            },
            "num_memories": num_memories,
            "mode": "measure",
            "source_position": "outer",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_4)


case_name = "P_sym_nocut_5mem"
num_parts = 64
max_iter = 10**5
num_parties = 4
base_params = {
    "P_LINK": 1,
    "F_INIT": 0.99,
    "T_P": 1e-6,  # preparation time, consistent with MHz entangled pair source
    "P_D": 1e-6,  # dark count probability
    "T_DP": 1,  # dephasing time
}
num_memories = 5
T_CUT = None  # cut off time
# mode = "distribute"
# source_position = "outer"


lengths = np.linspace(1e3, 220e3, num=num_parts)

case_specification_1 = {
    "name": case_name,
    "subcase_name": "distribute_central",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {
        part: {
            "distance_from_central": [lengths[part]] * int(num_parties - 1),
            "distance_A": lengths[part],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUT,
            },
            "num_memories": num_memories,
            "mode": "distribute",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_1)


case_specification_2 = {
    "name": case_name,
    "subcase_name": "distribute_outer",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {
        part: {
            "distance_from_central": [lengths[part]] * int(num_parties - 1),
            "distance_A": lengths[part],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUT,
            },
            "num_memories": num_memories,
            "mode": "distribute",
            "source_position": "outer",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_2)


case_specification_3 = {
    "name": case_name,
    "subcase_name": "measure_central",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {
        part: {
            "distance_from_central": [lengths[part]] * int(num_parties - 1),
            "distance_A": lengths[part],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUT,
            },
            "num_memories": num_memories,
            "mode": "measure",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_3)


case_specification_4 = {
    "name": case_name,
    "subcase_name": "measure_outer",
    "num_parts": num_parts,
    "index": lengths,
    "case_args": {
        part: {
            "distance_from_central": [lengths[part]] * int(num_parties - 1),
            "distance_A": lengths[part],
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUT,
            },
            "num_memories": num_memories,
            "mode": "measure",
            "source_position": "outer",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_4)

case_name = "P_sym150_per_TCUT_1mem"
num_parts = 64
max_iter = 10**5
num_parties = 4
length = 150e3
base_params = {
    "P_LINK": 1,
    "F_INIT": 0.99,
    "T_P": 1e-6,  # preparation time, consistent with MHz entangled pair source
    "P_D": 1e-6,  # dark count probability
    "T_DP": 1,  # dephasing time
}
num_memories = 1


T_CUTS = np.linspace(0.005, 0.5, num=num_parts)

case_specification_1 = {
    "name": case_name,
    "subcase_name": "distribute_central",
    "num_parts": num_parts,
    "index": T_CUTS,
    "case_args": {
        part: {
            "distance_from_central": [length] * int(num_parties - 1),
            "distance_A": length,
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUTS[part],
            },
            "num_memories": num_memories,
            "mode": "distribute",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_1)

case_specification_2 = {
    "name": case_name,
    "subcase_name": "distribute_outer",
    "num_parts": num_parts,
    "index": T_CUTS,
    "case_args": {
        part: {
            "distance_from_central": [length] * int(num_parties - 1),
            "distance_A": length,
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUTS[part],
            },
            "num_memories": num_memories,
            "mode": "distribute",
            "source_position": "outer",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_2)


case_specification_3 = {
    "name": case_name,
    "subcase_name": "measure_central",
    "num_parts": num_parts,
    "index": T_CUTS,
    "case_args": {
        part: {
            "distance_from_central": [length] * int(num_parties - 1),
            "distance_A": length,
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUTS[part],
            },
            "num_memories": num_memories,
            "mode": "measure",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_3)


case_specification_4 = {
    "name": case_name,
    "subcase_name": "measure_outer",
    "num_parts": num_parts,
    "index": T_CUTS,
    "case_args": {
        part: {
            "distance_from_central": [length] * int(num_parties - 1),
            "distance_A": length,
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUTS[part],
            },
            "num_memories": num_memories,
            "mode": "measure",
            "source_position": "outer",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_4)


### 2D case, new input distance from central: list of floats
case_name = "P_unis_T2_m_nocut"
num_parts_x = 20
num_parts_y = 20
num_parts = num_parts_x * num_parts_y
max_iter = 10**3
num_parties = 4
# distances between Düsseldorf and Köln, Wuppertal and Duisburg
distance_from_central = [31e3, 25e3, 27e3]
# distance between Düsseldorf and Siegen
distance_A = 76e3

T_CUT = None


ms = np.linspace(1, 20, num=num_parts_x)  # x-axis
T_2s = np.logspace(-3, 1, num=num_parts_y)  # y-axis

param_matrix = np.array(np.meshgrid(ms, T_2s)).T.reshape(-1, 2)


case_specification_1 = {
    "name": case_name,
    "subcase_name": "distribute_central",
    "num_parts": num_parts,
    "index": list(range(num_parts)),
    "case_args": {
        part: {
            "distance_from_central": distance_from_central,
            "distance_A": distance_A,
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": param_matrix[part][1],
                "T_CUT": T_CUT,
            },
            "num_memories": int(param_matrix[part][0]),
            "mode": "distribute",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}


cases.append(case_specification_1)

### 2D case, new input distance from central: list of floats
case_name = "P_70_N_m_nocut"
num_parts_x = 6
num_parts_y = 5
num_parts = num_parts_x * num_parts_y
max_iter = 10**2
# other distances
distance_B = 70e3
# distance to A link
distance_A = 70e3

T_CUT = None


Ns = np.linspace(4, 9, num=num_parts_x)  # x-axis
ms = np.linspace(1, 20, num=num_parts_y)  # y-axis

param_matrix = np.array(np.meshgrid(Ns, ms)).T.reshape(-1, 2)

case_specification_1 = {
    "name": case_name,
    "subcase_name": "distribute_central",
    "num_parts": num_parts,
    "index": list(range(num_parts)),
    "case_args": {
        part: {
            "distance_from_central": [distance_B] * int(param_matrix[part][0] - 1),
            "distance_A": distance_A,
            "num_parties": param_matrix[part][0],
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUT,
            },
            "num_memories": param_matrix[part][0],
            "mode": "distribute",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}


cases.append(case_specification_1)


case_name = "P_70_N_m_nocut_correct"
num_parts_x = 6
num_parts_y = 5
num_parts = num_parts_x * num_parts_y
max_iter = 10**2
# other distances
distance_B = 70e3
# distance to A link
distance_A = 70e3

T_CUT = None


Ns = np.linspace(4, 9, num=num_parts_x)  # x-axis
ms = np.linspace(4, 20, num=num_parts_y)  # y-axis

param_matrix = np.array(np.meshgrid(Ns, ms)).T.reshape(-1, 2)

case_specification_1 = {
    "name": case_name,
    "subcase_name": "distribute_central",
    "num_parts": num_parts,
    "index": list(range(num_parts)),
    "case_args": {
        part: {
            "distance_from_central": [distance_B] * int(param_matrix[part][0] - 1),
            "distance_A": distance_A,
            "num_parties": param_matrix[part][0],
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUT,
            },
            "num_memories": param_matrix[part][0],
            "mode": "distribute",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}


cases.append(case_specification_1)


case_name = "P_sym150_per_TCUT_1mem_v2"
num_parts = 64
max_iter = 10**4
num_parties = 4
length = 150e3
base_params = {
    "P_LINK": 1,
    "F_INIT": 0.99,
    "T_P": 1e-6,  # preparation time, consistent with MHz entangled pair source
    "P_D": 1e-6,  # dark count probability
    "T_DP": 1,  # dephasing time
}
num_memories = 1


T_CUTS = np.linspace(0.05, 2, num=num_parts)

case_specification_1 = {
    "name": case_name,
    "subcase_name": "distribute_central",
    "num_parts": num_parts,
    "index": T_CUTS,
    "case_args": {
        part: {
            "distance_from_central": [length] * int(num_parties - 1),
            "distance_A": length,
            "num_parties": num_parties,
            "max_iter": max_iter,
            "params": {
                "P_LINK": 1,
                "F_INIT": 0.99,
                "T_P": 1e-6,
                "P_D": 1e-6,
                "T_DP": 1,
                "T_CUT": T_CUTS[part],
            },
            "num_memories": num_memories,
            "mode": "distribute",
            "source_position": "central",
        }
        for part in range(num_parts)
    },
}
cases.append(case_specification_1)

num_cases = len(cases)

if __name__ == "__main__":
    start_idx = 0
    curr_name = cases[0]["name"]
    for idx, case_spec in enumerate(cases):
        if case_spec["name"] == curr_name:
            continue
        else:
            print(f"Case {curr_name} has case_numbers: {start_idx}-{idx-1}")
            start_idx = idx
            curr_name = case_spec["name"]
    # then print the last
    print(f"Case {curr_name} has case_numbers: {start_idx}-{num_cases - 1}")


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
