# Copyright 2019 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
import numpy as np
import os
import pandas as pd
import pickle
import pathlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from utilities import constants as CONSTANTS

src_path = str(pathlib.Path(__file__).parents[2])

class CPUestimator:

    CPU_INTERVAL_CATEGORY = {0: pd.Interval(-0.001, 20.0, closed='right'),
                            1: pd.Interval(20.0, 40.0, closed='right'),
                            2: pd.Interval(40.0, 60.0, closed='right'),
                            3: pd.Interval(60.0, 80.0, closed='right'),
                            4: pd.Interval(80.0, 100.0, closed='right')}
    def __init__(self):
        self.sq_duration = 2
        self.model = self.load_model('cpu_rfr_model.pkl')


    def predict_cpu(self, model_input):

        # Input is an array [ue_attach_rate_5m, ue_connected]

        pf_transform = PolynomialFeatures(degree=2)
        in_feat = pf_transform.fit_transform(model_input)

        pred = self.model.predict(in_feat)

        final_pred = self.sample_cpu_percentage(pred[0])

        return final_pred
        
    def __get_predicted_interval(self, predicted_class):

        pred_interval = self.CPU_INTERVAL_CATEGORY.get(predicted_class)

        return pred_interval

    def sample_cpu_percentage(self, predicted_class):    
        
        pred_interval = self.__get_predicted_interval(predicted_class)
        
        min, max = abs(pred_interval.left), abs(pred_interval.right)
       
        gamma_distrib = np.random.gamma(1.2, 0.8, size=(25, 1))
        gamma_distrib = np.tile(gamma_distrib, (1, self.sq_duration))

        mx_scaler = MinMaxScaler(feature_range=(min, max))
        scaled_distrib = mx_scaler.fit_transform(gamma_distrib)

        final_predicted_val = np.random.choice(scaled_distrib.flatten())
        
        return final_predicted_val
    
    @staticmethod
    def load_model(model_name):
        MODEL_DIR = 'models'
        model_path = os.path.join(src_path, MODEL_DIR, model_name)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        return model

#TODO Introduce spontaenity
class MemoryEstimator:

    MAX_FREE_RAM = 3.8*1024 #3Gib represented in Mib
    MIN_FREE_RAM = 1.5*1024 #1.5Gib
    MAX_MME_USAGE = 500
    MIN_MME_USAGE = 83


    def __init__(self):
        self.model = self.load_model('rfr_mme_model.pkl')
        self.mem_free_low_counter = 0
        self.mme_high_counter = 0
        self.mme_usage = self.MIN_MME_USAGE
        self.mem_free = self.MAX_FREE_RAM

    def get_mme_usage(self, model_inputs):

        self.mme_usage = self.model.predict(model_inputs)[0]
        #output is in Mib

        if self.mme_high_counter >= 5:
            self.mme_usage = self.MIN_MME_USAGE
            self.mme_high_counter = 0

        if self.mme_usage >= self.MAX_MME_USAGE:
            self.mme_high_counter +=1
            

        return self.mme_usage

    def get_ram_usage(self):

        if self.mem_free_low_counter >= 5:
            self.mem_free = self.MAX_FREE_RAM
            self.mem_free_low_counter = 0

        #Free memory = MAX_AVL_RAM - Current MME usage
        if (self.MAX_FREE_RAM - self.mme_usage) > self.MIN_FREE_RAM:
            return self.mem_free - self.mme_usage
        else:
            self.mem_free_low_counter+=1
            return self.MIN_FREE_RAM
        #Value returned is in Mib
        return self.mem_free

    @staticmethod
    def load_model(model_name):
        MODEL_DIR = 'models'
        model_path = os.path.join(src_path, MODEL_DIR, model_name)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        return model