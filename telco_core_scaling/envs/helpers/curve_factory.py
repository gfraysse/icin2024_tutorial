import math
import random
import numpy as np
from enum import Enum
from collections import defaultdict


class LoadType(Enum):
    WORKLOAD = 'load_path'
    NEW_ATTACHES = 'attach_rate_path'

class Curves:
    def _generate_sin():
        # 100 linearly spaced numbers
        x = np.linspace(100, 300, 215)
        x_array = []
        y = []
        for i in x:
            r = max(int(math.sin(2 * math.pi * i) * 800), 40)
            x_array.append(i)
            y.append(r)

        # print(y)

        y1 = np.array(y)
        load = np.tile(y1.reshape(-1,1),4).flatten()
        
        return x_array, load

    def sine_curve(step):
        _, y = Curves._generate_sin()
        r = y[step % len(y)] # this gives the total number of session for step
        # attach rate only considers new sessions between two steps
        # so except if we're at the first step, the current attach rate is the 
        # current number of sessions - the previous number of sessions
        # as this substraction can be negative, we return 0 for all negative values

        return r

    def sine_attach_rate(step, workload_history):
        history = list(workload_history)

        if step >= 3:
            r = max(10, (history[-1] - history[-2]))
            return r
        else:
            return 0
        
    def random_curve():
        return random.randint(0, 600)

    def custom_curve(step, workload):
        return workload[step]




class CurveFactory:
    METRICS = {
        'PRODUCTION_DATA': {
            'function': Curves.custom_curve,
            'parameters' : ['step', 'workload'],
            'options': {
                'load_path': 'telco_core_scaling/workload/production_100000_steps.npy',
                'attach_rate_path': 'telco_core_scaling/workload/production_attach_rate_100000_steps.npy'
            }
        },
        'EVALUATION_DATA': {
            'function': Curves.custom_curve,
            'parameters' : ['step', 'workload'],
            'options': {
                'load_path': 'telco_core_scaling/workload/evaluation_80000_steps.npy'
            }
        },
        'SINE_CURVE': {
            'function': Curves.sine_curve,
            'parameters' : ['step'],
            'options': {},
        },

        'SINE_ATTACH_RATE': {
            'function': Curves.sine_attach_rate,
            'parameters' : ['step', 'history'],
            'options': {},
        },
        'RANDOM': {
            'function': Curves.random_curve,
            'parameters' : [],
            'options': {},
        }
    }

    def  __init__(self, seletected_metric, load_type):
        metric_info =  self.METRICS[seletected_metric]
        
        self.func = metric_info['function']
        self.func_parameters = metric_info['parameters']
        
        options = defaultdict(str, metric_info['options'])
        if load_type == LoadType.WORKLOAD:
            if options['load_path']:
                self.workload = np.loadtxt(options['load_path']) 
            else:
                _, self.workload = Curves._generate_sin()
        else:
            self.workload = np.loadtxt(options['attach_rate_path']) if options['attach_rate_path'] else None
            
    def workload_size(self):
        return len(self.workload)
    
    def __select_parameters(self, step, workload, history):
        parameters = {
            'step' : step, 
            'workload' : workload, 
            'history' : history
        }
        
        return [parameters[f_par] for f_par in self.func_parameters]
    
    def get_metric(self, step, history=None): 
        parameters = self.__select_parameters(step, self.workload, history)
        return self.func(*parameters)