import yaml
import sys
import os
import pathlib
dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = str(pathlib.Path(dir).parents[0])

print("Reading the config file")
config_path = os.path.join(PROJECT_ROOT, 'config_per.yaml')
# with open('config_per.yaml', 'r') as f:
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
try:
    # MIN_EPSILON = config['MIN_EPSILON']
    # MAX_EPSILON = config['MAX_EPSILON']
    GAMMA = config['GAMMA']

    INITIAL_BETA = config['INITIAL_BETA']
    BETA_INC = config['BETA_INC']
    PER_ALPHA = config["PER_ALPHA"]

    ACTION_SEED = config['ACTION_SEED']
    BATCH = config['BATCH']
    LOAD_BUFFER = config['LOAD_BUFFER']

    EPOCHS = config['EPOCHS']
    METRICS_LIST = config['METRICS_LIST']

    RESUME_TRAINING = config['RESUME_TRAINING']
    OPTIM = config['OPTIM']
    REPLAY_MEMORY = config['REPLAY_MEMORY']
    INITIAL_EPSILON = config['INITIAL_EPSILON']
    FINAL_EPSILON = config['FINAL_EPSILON']
    EXPLORE = config['EXPLORE']
    TARGET_UPDATE_STEPS = config['TARGET_UPDATE_STEPS']
    INITIAL_LR = float(config['INITIAL_LR'])

    #SAVED_MODELS_PATH = config['SAVED_MODELS_PATH']
    MIN_BUFFER_SIZE_FOR_TRAIN = config['MIN_BUFFER_SIZE_FOR_TRAIN']
    BUFFER_NAME = config['BUFFER_NAME']
    STEPS = config['STEPS']
    MAX_UE_CONNECTED = config['MAX_UE_CONNECTED']

    PROJECT_NAME = config['PROJECT_NAME']
    
    BACKUPS_PATH = os.path.join(PROJECT_ROOT, config["BACKUPS_PATH"])
    SAVED_MODELS_PATH = os.path.join(PROJECT_ROOT, config["SAVED_MODELS_PATH"])
    SAVED_BUFFER_PATH = os.path.join(PROJECT_ROOT, config["SAVED_BUFFER_PATH"])
    WORKLOAD_LOGS_PATH = os.path.join(PROJECT_ROOT, config["WORKLOAD_LOGS_PATH"])
    TENSORBOARD_ROOT_DIR = os.path.join(PROJECT_ROOT, config["TENSORBOARD_ROOT_DIR"])
    TENSORBOARD_RUNS_DIR = os.path.join(PROJECT_ROOT, config["TENSORBOARD_RUNS_DIR"])

    MODEL_CONFIGS = {
        "GAMMA": GAMMA,

        "INITIAL_BETA" : INITIAL_BETA,
        "BETA_INC" : BETA_INC,
        "PER_ALPHA" : PER_ALPHA,

        "BATCH" : BATCH,

        "EPOCHS" : EPOCHS,

        "RESUME_TRAINING" : RESUME_TRAINING,
        "REPLAY_MEMORY" : REPLAY_MEMORY,
        "INITIAL_EPSILON" : INITIAL_EPSILON,
        "FINAL_EPSILON" : FINAL_EPSILON,
        "EXPLORE" : EXPLORE,
        "TARGET_UPDATE_STEPS" : TARGET_UPDATE_STEPS,
        "INITIAL_LR" : INITIAL_LR,

        #SAVED_MODELS_PATH : config['SAVED_MODELS_PATH']
        "MIN_BUFFER_SIZE_FOR_TRAIN" : MIN_BUFFER_SIZE_FOR_TRAIN,
        "STEPS" : STEPS
    }
    
except KeyError as e:
    print(f"The parameter {str(e)} is not present in the config file, please double check")
    sys.exit("Exiting due to some missing parameters")