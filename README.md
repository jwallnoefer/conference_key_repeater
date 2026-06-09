# Strategy optimisation for quantum conference key agreement in asymmetric star networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20085326.svg)](https://doi.org/10.5281/zenodo.20085326)

This repository is an archive for the code used in:

> Strategy optimisation for quantum conference key agreement in asymmetric star networks <br>
> J. Memmen, J. Kunzelmann, J. Wallnöfer, N. Walk, J. Eisert <br>
> Preprint: [arXiv:2605.18677 [quant-ph]](https://doi.org/10.48550/arXiv.2605.18677)

## Goal
Use [ReQuSim](https://github.com/jwallnoefer/requsim), a simulator for quantum repeater protocols,
to simulate the distribution of GHZ states for a quantum conference agreement protocol.

## Repository structure

The main parts of the repository are structured as follows:

* The `tools` package that includes:
  - A custom `ReQuSim`-event for connecting N Bell pairs to an N-qubit GHZ state
  - Various helper functions to set up the simulation and evaluate their output
* The `scenarios/multi_memory` directory with
  - The general simulation scenario with all options in `multi_memory.py`
  - A definition of all parameters for all the cases we used in `case_definition.py`

As minor parts it also includes tests for the new `ReQuSim` event and the `run_tools`
for quickly performing simulations that are specified in the case_definition format we used
as well as launching these jobs on a HPC system with SLURM (some tweaks would probably be
necessary to reuse).

### How to use

To run the simulations you need to have the `requsim` package installed. For performance, we
used some in-development features, so you will need to install a version of at least 0.5dev67
or higher, for example to install this particular version use:
```bash
pip install git+https://github.com/jwallnoefer/requsim@e55a089b8ed68f0a6153516d0e633fcaa46113b6#egg=requsim
```
In order to use the run tools `docopt` is needed. If you wish to use the plotting files,
`matplotlib` is also required.

For best results we recommend recreating the same virtual environment we used to develop
the code via pipenv. This assumes you have a Python>=3.9 and `pipenv` installed on your system:

```bash
pipenv sync --dev
```

You can activate the environment with
```bash
pipenv shell
```
(or, alternatively, run commands in the environment with `pipenv run COMMAND`).
