# from Pipeline  reward_wrapper.py
import logging
logger = logging.getLogger('reward_func')

log_format = "%(asctime)s - %(levelname)s - %(message)s"
level = logging.INFO
#filename = 'environment.log'
logging.basicConfig(level=level, format=log_format)

def reward_function(next_state, done):
    reward_c = 0
    logger.info(f"reward_state_dict: {next_state}, {done}")
    if isinstance(next_state, dict):
        if not done:
            try:
                if next_state['num_calls_dropped'] > 0:
                    reward_c  = - next_state['num_calls_dropped']
                elif next_state['magma_mme_service']  <= 0.7:
                    reward_c = 1 - abs(0.7 - next_state['magma_mme_service'])
                else:
                    reward_c = - next_state['magma_mme_service'] * 2
                if reward_c < -10:
                    reward_c = -10
            except KeyError as c :
                logger.info(f"key {c} is missing")
        else:
            logger.info("Penalty for touching boundary")
            reward_c = -1
    return reward_c