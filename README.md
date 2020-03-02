# torch-template
Boiler plate code for Torch based ML projects

### Dev Setup

* Works with Python 3.6, 3.7, 3.8
* `pip install -r requirements.txt`
* Install pre-commit hooks `pre-commit install`
* The code is linted using:
    * `black`
    * `flake8`
    * `mypy`
* Lint tests can be run locally using `nox -s lint`
* The entry point for the code is `codes/app/main.py`
* Run the code using `PYTHONPATH=. python codes/app/main.py`. This uses the config in `config/sample_config.yaml`.
* To run the code with a different config, make a copy of the config file in the `config` directory. Then run `PYTHONPATH=. python codes/app/main.py --config_id custom_config_name.yaml`.

### Acknowledgements

* Config for `circleci`, `pre-commit`, `mypy` etc are borrowed/modified from [Hydra](https://github.com/facebookresearch/hydra)


## Parsing Logs

* The generated log file contains a json string per line. It can be grepped using [jq](https://stedolan.github.io/jq/).
* For example, to grep the `mode` and the `loss` fields in the log files, run `cat log.jsonl | jq --compact-output '[.mode, .loss]'`.
