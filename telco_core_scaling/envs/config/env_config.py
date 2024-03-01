import yaml
import sys
import os

ENV_ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_CONFIG_FILE = 'config_env.yaml'

config_path = os.path.join(ENV_ROOT, ENV_CONFIG_FILE)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
try:
    SEED = config["SEED"]
    ENV_INFO = config["ENV_INFO"]
    MAX_ENV_INFO = config["MAX_ENV_INFO"]
    EVALUATION_METRICS = config["EVALUATION_METRICS"]
    OBSERVATION_METRICS = config["OBSERVATION_METRICS"]
    ENV_DEFAULT_OPTIONS = config["ENV_DEFAULT_OPTIONS"]
    
except KeyError as e:
    print(f"The parameter {str(e)} is not present in the config file, please double check")
    sys.exit("Exiting due to some missing parameters")