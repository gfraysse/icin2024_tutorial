import collections
import json
import numpy as np
from utilities import constants as CONSTANTS

class Instance: # TODO: transform in Interface

    def __init__(self, instance_id, ue_connected, current_attach):
        # parametrize variables de instance
        self.instance_id = instance_id
        self.instance_attach_history = collections.deque(maxlen=300)
        self.new_attaches = current_attach

        self.max_ue_connected = CONSTANTS.MAX_UE_CONNECTED
        self.max_ue_attach_rate = 3.03
        self.num_calls_dropped = 0

        self.set_connected_user(ue_connected)
        self.__update_attach_history()

    def calculate_attach_rate(self):
        if len(self.instance_attach_history) < 20: #For sine wave workload
            self.ue_attach_rate = (self.instance_attach_history[-1] - np.min(self.instance_attach_history)) / (15 * len(self.instance_attach_history)) #For sine wave workload

        else:

            self.ue_attach_rate = (self.instance_attach_history[-1] - list(self.instance_attach_history)[-20]) / (60*5)

        if self.ue_attach_rate > self.max_ue_attach_rate:
            self.ue_attach_rate = self.max_ue_attach_rate
    
    def update_instance(self, ue_connected, new_attaches):

        
        #update new users attached
        self.new_attaches = new_attaches

        self.set_connected_user(ue_connected)
        
        
        self.__update_attach_history()

        #Update the attach rate for the VM here
        self.calculate_attach_rate()

        # print("Instance attach rate history", self.instance_attach_history)

    def __update_attach_history(self):
        if not len(self.instance_attach_history):
            cum_attach = self.new_attaches
        else:
            cum_attach = self.new_attaches + list(self.instance_attach_history)[-1]
        
        self.instance_attach_history.append(cum_attach)

    def get_id(self):
        return self.instance_id

    def set_connected_user(self, ue_connected):

        if ue_connected > self.max_ue_connected:
            self.num_calls_dropped = ue_connected - self.max_ue_connected
            self.ue_connected = self.max_ue_connected
        else:
            self.num_calls_dropped = 0
            self.ue_connected = ue_connected
    
    def get_cpu_usage(self):
        return self.cpu_percentage
    
    def set_cpu_usage(self, cpu_percent):
        self.cpu_percentage = cpu_percent
    
    def get_mem_free(self):
        return self.mem_free

    def set_mem_free(self, mem_free):
        self.mem_free = mem_free
    
    def get_mme_usage(self):
        return self.magma_mme_service
    
    def set_mme_usage(self, mme_usage):
        self.magma_mme_service = mme_usage
    
    def get_connected_users(self):
        return self.ue_connected
    
    def get_attach_rate(self):
        return self.ue_attach_rate
    
    def get_nb_calls_dropped(self):
        return self.num_calls_dropped

    def __str__(self):

        instance_info = {
                'ue_connected': self.get_connected_users(),
                'ue_attach_rate': self.get_attach_rate(),
                'cpu_percentage': self.get_cpu_usage(),
                'mem_free': self.get_mem_free(),
                'magma_mme_service': self.get_mme_usage(),
                'num_calls_dropped': self.get_nb_calls_dropped()
            }
        
        return f"Values for this instance are:\n{json.dumps(instance_info, indent=4)}"

    def __repr__(self):

        return f"Instance({self.get_connected_users}, {self.new_attaches})"