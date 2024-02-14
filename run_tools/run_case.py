"""Run a specific part of a case in a requsim simulation or collect results.

Usage: run_case.py PATH CASE PART [--results=RESULTPATH] [--runexisting]
       run_case.py PATH CASE --collect [--results=RESULTPATH]
       run_case.py (-h | --help)

Arguments:
    PATH    path to the scenario directory, which must contain case_definition.py
    CASE    case number
    PART    which part of the case to run

Options:
    -h --help               show this help message
    --results=RESULTPATH    specify the result directory explicitly, will be inferred from PATH and CASE otherwise
    --runexisting           run the part even if it already exists
    --collect               collect the results instead of running
"""

from docopt import docopt
import sys
import os

# from requsim.tools.evaluation import standard_bipartite_evaluation
from common.evaluation import standard_ghz_evaluation
import pandas as pd
from datetime import datetime


if __name__ == "__main__":
    args = docopt(__doc__)

    sys.path.append(os.path.abspath(args["PATH"]))
    import case_definition as cd

    result_path = args["--results"]
    case = int(args["CASE"])
    if result_path is None:
        result_path = os.path.join(
            "results", cd.scenario_name, cd.name(case), cd.subcase_name(case)
        )

    if args["--collect"]:
        base_index = cd.index(case)
        index = []
        results = pd.DataFrame()
        for part in range(cd.num_parts(case)):
            try:
                results = pd.concat(
                    [
                        results,
                        pd.read_csv(
                            os.path.join(result_path, "parts", f"part{part}.csv")
                        ),
                    ]
                )
            except FileNotFoundError:
                print(f"part{part} not found in collect")
                continue
            index.append(base_index[part])
        results.index = index
        results.to_csv(os.path.join(result_path, "result.csv"))
    else:
        part = int(args["PART"])
        output_path = os.path.join(result_path, "parts")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not args["--runexisting"] and os.path.exists(
            os.path.join(output_path, f"part{part}.csv")
        ):
            print(
                f"Skipping part{part} because it already exists. Use option --runexisting to run anyway."
            )
        else:
            run_func = cd.run_func(case=case, part=part)
            run_args = cd.case_args(case=case, part=part)
            with open(os.path.join(output_path, f"part{part}.log"), "w") as f:
                start_time = datetime.now()
                print(f"Start time: {start_time}", file=f)
                print(run_args, file=f)
            p_return = run_func(**run_args)
            res = p_return.data
            res.to_pickle(os.path.join(output_path, f"part{part}.bz2"))
            evaluated_res = pd.DataFrame(
                [standard_ghz_evaluation(res)],
                columns=[
                    "raw_rate",
                    "fidelity",
                    "fidelity_err",
                    "key_per_time",
                    "key_per_time_err",
                ],
            )
            evaluated_res.to_csv(
                os.path.join(output_path, f"part{part}.csv"), index=False
            )
            # some protocols can collect extra data
            try:
                extra_data = p_return.extra_data
            except AttributeError:
                extra_data = None
            if extra_data is not None:
                extra_data.to_pickle(os.path.join(output_path, f"part{part}_extra.bz2"))

            with open(os.path.join(output_path, f"part{part}.log"), "a") as f:
                end_time = datetime.now()
                print(f"End time: {end_time}", file=f)
                print(f"Time elapsed: {end_time - start_time}", file=f)
