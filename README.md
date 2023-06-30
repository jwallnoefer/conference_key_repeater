# Distributing GHZ states for Conference Key Agreement

## Goal
Use [ReQuSim](https://github.com/jwallnoefer/requsim) to simulate the distribution of GHZ
states for a quantum conference agreement protocol.

## Dev Environment

We use `pipenv` to ensure a stable dev environment across
devices and contributors. (it is horribly slow, but it gets the job done.)

First, install `pipenv`:
```
pip install pipenv
```

Then, set up the new virtual environment:
```
pipenv sync --dev
pipenv run pre-commit install
```

You can activate the environment with
```
pipenv shell
```
(or, alternatively, run commands in the environment with `pipenv run COMMAND`).
