import collections
import json
import random
import sys

import gymnasium as gym
from gymnasium import spaces
import math
import numpy as np
import pandas as pd

from overrides import overrides

from .config.env_config import *
from .helpers.curve_factory import CurveFactory, LoadType

from .rendering.renderer import TelcoCoreScalingRenderer
from .helpers.instance import Instance
from .helpers.estimators import MemoryEstimator, CPUestimator
from .helpers.reward_wrapper import reward_function

import logging


logger = logging.getLogger('TelcoCoreScalingEnv')
log_format = "%(asctime)s - %(levelname)s - %(message)s"
level = logging.debug
logging.basicConfig(level=level, format=log_format)

RENDERING_ENABLED = False

class TelcoCoreScalingState():
    def __init__(self):
        self.info = ENV_INFO
        self.max_values = MAX_ENV_INFO

# For common interface for all attach rates
# A method can return the number of attaches 
# Instance class will handle the calculation of attach rate 
# ( Maintain counter of attaches & calculate attach rate)


class TelcoCoreScalingEnv(gym.Env):
    def __all_seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    @overrides
    def __init__(self, *args, **kwargs):
        self.__all_seed(seed = SEED)

        self.env_info = TelcoCoreScalingState()
        if RENDERING_ENABLED:
            
            self.renderer = TelcoCoreScalingRenderer()
        else:
            self.renderer = None

        keys = OBSERVATION_METRICS
        self.__set_observation_keys(keys)
        
        self.scaling_env_options = ENV_DEFAULT_OPTIONS
        self.sim_size = ENV_DEFAULT_OPTIONS['size']

        self.load_curve = CurveFactory(ENV_DEFAULT_OPTIONS['metric'], 
                                       LoadType.WORKLOAD)
        
        self.attach_curve = CurveFactory(ENV_DEFAULT_OPTIONS['attach_metric'], 
                                       LoadType.NEW_ATTACHES)

        self.actions = ENV_DEFAULT_OPTIONS['discrete_actions']
        self.num_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.num_actions)
        
        self.observation_size = ENV_DEFAULT_OPTIONS['observation_size']

        self.observation_space = spaces.Box(low = 0.0,
                                            high = sys.float_info.max,
                                            shape = (1, 5))
        
        self.max_instances = ENV_DEFAULT_OPTIONS['max_instances']
        self.min_instances = ENV_DEFAULT_OPTIONS['min_instances']
        self.capacity_per_instance = ENV_DEFAULT_OPTIONS["capacity_per_instance"]

        self.offset = ENV_DEFAULT_OPTIONS['offset']
        self.sim_size = ENV_DEFAULT_OPTIONS['size']
        self.change_rate = ENV_DEFAULT_OPTIONS['change_rate']
        self.max_history = math.ceil(self.sim_size[0])

        self.safe_env = ENV_DEFAULT_OPTIONS['safe_env']

        self.cpu_estimator = CPUestimator()
        self.memory_estimator = MemoryEstimator()

        self.cum_ins = 0
        self.step_idx = 0
        self.total_cost = 0.0

        self.load_history = collections.deque(maxlen = self.max_history)

        super().__init__(*args, **kwargs)

    def get_limits(self):
        return self.env_info.max_values
    
    def get_action_space(self):
        return self.actions
        
    def __compute_load(self):
        self.env_info.info['load'] = self.load_curve.get_metric(self.step_idx)
        self.load_history.append(self.env_info.info['load'])
        self.env_info.info['new_attaches'] = self.attach_curve.get_metric(self.step_idx, self.load_history)
    


    def __set_observation_keys(self, keys):
        self.observation_keys = sorted(keys)
        
    def __step_fwd(self):

        if self.step_idx > self.load_curve.workload_size():
            self.step_idx = 0
        else:
            self.step_idx += 1

    @overrides
    def reset(self, seed=42, options=None):

        super().reset(seed=seed)
        print("RESET")


        #Set all the values to default
        self.__step_fwd()
        self.instances = []
        self.num_instances = 0
        self.crash_probability = 0


        self.__compute_load()

        self.__do_action(1)

        self.__compute_state()

        observation = self.__get_observation()

        logger.info("\nState after reset: \n%s", json.dumps(observation, indent=4))

        return np.fromiter(observation.values(), dtype='float32'), self.env_info.info


    @overrides
    def step(self, action):

        #Increment the timestep
        self.__step_fwd()

        #Update the state
        self.prev_state = self.next_state

        #Compute the load
        self.__compute_load()

        #Execute the given action
        self.__do_action(action)

        #Compute the state after executing the action
        self.__compute_state()

        print(f"Number of VM: {self.num_instances}")
        
        
        #Check if environment is crashed
        crashed = self.__is_crashed()

        if crashed:
            self.env_info.info['nb_crash'] += 1

        #If set the value for truncated 'indicated whether environment is active or crashed
        terminated = False
        truncated = crashed

        logger.info("\nNext State: \n%s", json.dumps(self.next_state, indent=4))

        #Calculate the reward using the next state and crash status
        reward = reward_function(self.next_state, truncated)

        #Get a final observation array that will be the input to the agent
        observation = self.__get_observation()

        return np.fromiter(observation.values(), dtype='float32'), reward, terminated, truncated, self.env_info.info

    def __get_observation(self):

        final_state = collections.defaultdict(dict)
        metrics_list = OBSERVATION_METRICS

        for metric in sorted(metrics_list):
            final_state[metric] = self.next_state[metric]
        
        return final_state


    def __is_crashed(self):
        # compute crash probability, it increases as the attach rate is high
        crashed = False
        current_attach_rate = self.next_state['ue_attach_rate'] * self.env_info.max_values['ue_attach_rate']

        if current_attach_rate > 2.6:
            self.crash_probability += self.env_info.max_values['ue_attach_rate'] - current_attach_rate

        if self.crash_probability > 1:
            self.crash_probability = 1

        if self.crash_probability != 0:
            r = random.random()
            crash_at_this_step = (r <= self.crash_probability)
            if crash_at_this_step == True or self.crash_probability == 1:
                print("CRASH", self.crash_probability, current_attach_rate, r)
                crashed = True

        return crashed
    
    def __render_frame(self):
        self.renderer.render_env(self)
            
    @overrides
    def render(self, mode='human'):
        if RENDERING_ENABLED == True:
            self.__render_frame()
    
    @overrides
    def close(self):
        pass

    def __compute_state(self):

        self.next_state = self.__aggregate_metrics()
        print("Length of instances", len(self.instances))

        self.next_state['vm_count'] = len(self.instances) / self.env_info.max_values['vm_count']

    def __load_stats_per_vm(self):
        """Method to calculate ue_connected and new attaches per VM"""

        per_vm_ue = self.load_history[-1] / self.num_instances

        per_vm_attaches = self.env_info.info['new_attaches'] / self.num_instances

        return per_vm_ue, per_vm_attaches
        
    def __compute_cpu(self, model_inputs):

        cpu_usage = self.cpu_estimator.predict_cpu(model_inputs)
        return cpu_usage
    
    def __compute_mme_usage(self, model_inputs):

        mme_consumption = self.memory_estimator.get_mme_usage(model_inputs)
        return mme_consumption
    
    def __compute_mem_free(self):

        ram_usage = self.memory_estimator.get_ram_usage()
        return ram_usage
    
    def __prepare_inputs(self, in_array):

        model_inputs = pd.DataFrame(data=in_array, columns=['ue_attach_rate_5m', 'ue_connected'])

        return model_inputs


    def __operate_instance(self):

        for instance in self.instances:
            
            connected_users, new_attaches = self.__load_stats_per_vm()
            instance.update_instance(connected_users, new_attaches)

            in_array = np.array([
                [instance.get_attach_rate(), instance.get_connected_users()]
            ])

            model_inputs = self.__prepare_inputs(in_array)

            cpu_usage = self.__compute_cpu(model_inputs)
            mme_consumption = self.__compute_mme_usage(model_inputs)
            mem_free = self.__compute_mem_free()

            self.__set_instance_metrics(instance, cpu_percentage=cpu_usage, mme_consumption=mme_consumption, mem_free=mem_free)

            print(str(instance))
    
    def __do_action(self, action_idx):
        action = self.actions[action_idx]

        logger.info("Executed Action => %s" %(str(action)))

        new_instances = len(self.instances) + action
        
        if self.max_instances >= new_instances >= self.min_instances:

            self.num_instances = new_instances
            # Compute number of users each Instance is going to have and then initialize list of Instances with same UEs
            per_vm_ue, per_vm_attach = self.__load_stats_per_vm()

            if action > 0:
                instances_id = [k+1 for k in range(len(self.instances), (len(self.instances) + action))]

                for k in range(action):
                    self.instances.append(
                        Instance(instances_id[k], per_vm_ue, per_vm_attach))

            if action < 0:
                for _ in range(-1 * action):
                    self.instances.pop(0)

        #Regardless of the appending Instance() object in the list, each existing object should also be updated accordingly
        self.__operate_instance()

    def __set_instance_metrics(self, instance, cpu_percentage, mem_free, mme_consumption):
        """Method to set attributes of an Instance"""

        instance.set_cpu_usage(cpu_percentage)
        instance.set_mem_free(mem_free)
        instance.set_mme_usage(mme_consumption)

    def __aggregate_metrics(self):
        """Method to aggregate and normalize the metric values"""

        metrics_list = EVALUATION_METRICS

        state_info = collections.defaultdict(dict)
        for metric in metrics_list:
            val = 0
            norm_in_val = 0
            for instance in self.instances:
                val = getattr(instance, metric)
                # Sum the values across instances
                norm_in_val += self.__normalize_metric(val, metric)

            # Store the aggregated value for the metric in a dictionary
            state_info[metric] = norm_in_val

        return state_info

    def __normalize_metric(self, val, metric):
        max_val = self.env_info.max_values[metric]
        normalized_val = val / (max_val * len(self.instances))
        # normalized_val = val / max_val
        if metric in ['num_calls_dropped']:
            return val / max_val
        # if metric not in ['ue_attach_rate', 'num_calls_dropped']:
        #         return normalized_val / len(self.instances)
        
        return normalized_val