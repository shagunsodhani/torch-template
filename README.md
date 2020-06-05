[![CircleCI](https://circleci.com/gh/shagunsodhani/torch-template.svg?style=svg)](https://circleci.com/gh/shagunsodhani/torch-template) ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

# torch-template
Boiler plate code for Torch based ML projects

### Dev Setup

* Works with Python 3.6, 3.7, 3.8
* `pip install -r requirements.txt`.
* Install pre-commit hooks `pre-commit install`.
* The code is linted using:
    * `black`
    * `flake8`
    * `mypy`
* Lint tests can be run locally using `nox -s lint`.
* Mpyp tests can be run locally using `nox -s mpyp`.
* The entry point for the code is `main.py`.
* A sample jupyter notebook is available in `codes/notebook`.
* Run the code using `PYTHONPATH=. python main.py`. This uses the config in `config/config.yaml`.
* The code uses [Hydra] (https://github.com/facebookresearch/hydra) framework for composing configs.

### Acknowledgements

* Config for `circleci`, `pre-commit`, `mypy` etc are borrowed/modified from [Hydra](https://github.com/facebookresearch/hydra)


## Parsing Logs

* The generated log file contains a json string per line. It can be grepped using [jq](https://stedolan.github.io/jq/).
* For example, to grep the `mode` and the `loss` fields in the log files, run `cat log.jsonl | jq --compact-output '[.mode, .loss]'`.
