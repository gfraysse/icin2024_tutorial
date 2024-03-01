"""
 The rendering of the live animation is created using Flask
 It uses the templates/render.html page which load the images
 in the "img/" folder incrementally according to their names: img_1.jpg then img_2.jpg
"""
import os
import math
import random
import cv2
#from flask import Flask, render_template, Response
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing   

# to remove warning

# SettingWithCopyWarning: 
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead
# See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

class TelcoCoreScalingRenderer():
    def __init__(self):        
        pd.options.mode.chained_assignment = None  # default='warn'

        # The images are generated in IMG_FOLDER, it is also used as the URL to access them statically with Flask\
        
        self.IMG_FOLDER ="img/"
        self.AGENT_LOGO = cv2.imread('rl-agent-logo2.jpg')
        print("AGENT LOGO SHAPE", self.AGENT_LOGO.shape)
        self.AGENT_LOGO = cv2.resize(self.AGENT_LOGO, (100,100))

        self.img_array = []

        if not os.path.exists(self.IMG_FOLDER):
            os.mkdir(self.IMG_FOLDER)
        # webapp = Flask(__name__,
        #                static_url_path = "/" + IMG_FOLDER, 
        #                static_folder = IMG_FOLDER,
        #                template_folder='templates')

        self.ACTIONS = [-1, 0, 1]
        self.REWARD_PAST_VALS = []
        self.LOAD_PAST_VALS = []
        self.ACTION_PAST_VALS = []

        self.HEIGHT = 720
        self.WIDTH = 1280
        # self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT = cv2.FONT_HERSHEY_TRIPLEX

        # DURATION = 2000

        self.WRITE_IMGS = True

    """
    Private functions
    """
    #GF updated so that it returns only the last line of the log file in df, and directly the history of the last 10 values of 'action' and 'load' 
    def __get_last_state(self, history_len = 10):
        state_logs = pd.read_csv('state_logs_2023_05_30_11_37_08.csv')
        combined_logs = state_logs[state_logs.vm_name=='combined'] #SV for -ve reward
        combined_logs['vm_count'] *= 5
        combined_logs.reset_index(inplace = True, drop = True)
        # actions_history = combined_logs[combined_logs.index > combined_logs.shape[0] - history_len]['action']
        actions_history = combined_logs['action']
        # load_history = combined_logs[combined_logs.index > combined_logs.shape[0] - history_len]['load']
        load_history = []
        #GF why a deep copy below ? it can get costly when the DataFrame size increases
        # df = combined_logs.loc[combined_logs.index[-1], :] # .copy(deep=True)
        #SV reading complete file for debugging
        df = combined_logs.loc[combined_logs.index[:-1], :] # .copy(deep=True)
        return df, actions_history, load_history, state_logs

    """
    To generate dummy data
    """
    def generate_sin():
        # 100 linearly spaced numbers
        x = np.linspace(0, 1000, 1000)
        x_array = []
        y = []
        for i in x:
            r = r = max(int(math.sin(2*math.pi*i) * 500), 0)
            x_array.append(i)
            y.append(r)
            
            # print(y)
            
            return x_array, y

    """
    To generate dummy data
    """
    def __generate_pseudo_data(self, duration):
        random.seed(42)
    
        df = pd.DataFrame()
        t_0 = 1686832966
        nbVM = 1
        x_load, y_load = generate_sin()
        for i in range(duration):
            action = random.randrange(3)
            nbVM += self.ACTIONS[action]
            reward = random.random()
        
            if nbVM < 0:
                nbVM = 0
            if nbVM > 5:
                nbVM = 5

            datapoint = pd.DataFrame(data = {
                't': t_0 + i,
                'nbVM': nbVM,
                'load': y_load[i % len(y_load)],
                'action': self.ACTIONS[action],
                'reward': reward,
            }, index = [i])
            
            df = pd.concat([df, datapoint])
        
        return df 
                  
    def __get_seg_data(self, state_logs):
        """Preprocess the dataframe to extract metrics from individual VM"""

        # I think the copy is useless and might become costly as the file grow longer
        raw_logs = state_logs[state_logs.vm_name != 'combined'] #.copy(deep=True)
        
        raw_logs.reset_index(drop = True, inplace = True)
        
        lb_enc = preprocessing.LabelEncoder()
        
        raw_logs['new_idx'] = lb_enc.fit_transform(raw_logs['timestamp'])
        
        raw_logs['chain_id'] = raw_logs['vm_name'].apply(lambda x: int(x.split('-')[0][-1]))
        
        grouped_logs = raw_logs.groupby('new_idx')
    
        return grouped_logs

    def __get_feature(self, grouped_df, t, chain_id, feature):
        single_df = grouped_df.get_group(t)
    
        ue_count = single_df[single_df['chain_id'] == chain_id][feature].fillna(0)
        if len(ue_count) == 0:
            return 0
        return ue_count.to_numpy('int64')[0]

    def __get_total_ue(self, grouped_df, t, feature):
        total_ue = grouped_df.get_group(t).fillna(0).sum()[feature]
    
        # ue_count = single_df[single_df['chain_id'] == chain_id][feature].fillna(0)
        # if len(ue_count) == 0:
        #     return 0
        # return ue_count.to_numpy('int64')[0]

        return int(total_ue)
    

    def __get_ue_count(self, grouped_df, t, chain_id):
        """Method to get UE count from VMs via chain_id"""

        return self.__get_feature(grouped_df, t, chain_id, 'ue_connected')

    def __get_cpu(self, grouped_df, t, chain_id):
        """Method to get CPU usage from VMs via chain_id"""

        return self.__get_feature(grouped_df, t, chain_id, 'cpu_percentage')

    def __get_mem(self, grouped_df, t, chain_id):
        """Method to get memory usage from VMs via chain_id"""
        
        return self.__get_feature(grouped_df, t, chain_id, 'magma_mme_service')

    def __draw_tester(self, img, x, y, chain_id):
        # for rectangle: top-left corner and bottom-right , then RGB color
        # cv2.rectangle(img,(x, y), (x + 250, y - 80), (255, 0, 0), 0)
        cv2.rectangle(img,(x, y), (x + 250, y - 80), (0,95, 255), -1)
        
        # coordinate is bottom-left corner where data starts
        # cv2.putText(img, 'S1APTester #' + str(chain_id), (x + 10, y - 20), self.FONT, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'S1APTester #' + str(chain_id), (x + 40, y - 30), self.FONT, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        
    def __draw_AGW(self, img, x, y, chain_id, grouped_vm, t):
        # for rectangle: top-left corner and bottom-right , then RGB color
        # cv2.rectangle(img,(x, y), (x + 200, y - 80), (255, 0, 0), 0)
        cv2.rectangle(img,(x, y), (x + 200, y - 80), (0,95,255), -1)
        
        # coordinate is bottom-left corner where data starts
        # cv2.putText(img, 'AGW #' + str(chain_id), (x + 40, y - 20), self.FONT, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        cv2.putText(img, 'AGW #' + str(chain_id), (x + 55, y - 30), self.FONT, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        
        # display CPU/mem
        cpu = self.__get_cpu(grouped_vm, t, chain_id)
        mem = self.__get_mem(grouped_vm, t, chain_id)
        cv2.putText(img, f"CPU: {cpu} %", (x + 20, y - 60), self.FONT, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"MEM: {mem} MB", (x + 100, y - 60), self.FONT, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    def __draw_chain(self, img, chain_id, t, grouped_vm):
        x_start = 300
        self.__draw_workload_gen(img, chain_id)
        self.__display_ue_count(img, grouped_vm, t, chain_id)
        self.__draw_tester(img, x_start, self.HEIGHT - 20 - 100 * (chain_id - 1), chain_id)
        self.__draw_AGW(img, x_start + 300, self.HEIGHT - 20 - 100 * (chain_id - 1), chain_id, grouped_vm, t)
        cv2.line(img, (x_start + 250, self.HEIGHT - 50 - 100 * (chain_id - 1)), (x_start + 300, self.HEIGHT- 50 - 100 * (chain_id - 1)), (0, 0, 0), 1)
        
        cv2.rectangle(img,(590, 140), (590+250, 190), (0,95, 255), 1)
        cv2.putText(img, "MAGMA (LTE Core NF)", (600,170), self.FONT, 0.6, (255, 0, 0), 1, cv2.LINE_AA)    

    def __display_ue_count(self,img, grouped_logs, t, chain_id):
            
        x_start = 30
        #y_start = self.HEIGHT-100
        
        ue_count = self.__get_ue_count(grouped_logs, t, chain_id)
        cv2.putText(img, f"{ue_count} UEs", (x_start + 185, self.HEIGHT- 115 - 80 * (chain_id - 1)), self.FONT, 0.525, (255, 0, 0), 1, cv2.LINE_AA)    
        
    def __draw_workload_gen(self, img, chain_id):
        x_start = 30
        y_start = self.HEIGHT-100
        # cv2.rectangle(img,(x_start, y_start), (x_start + 75, y_start - 300), (255, 0, 0), 0)
        cv2.rectangle(img,(x_start, y_start), (x_start + 75, y_start - 300), (0, 0, 0), -1)
        # cv2.putText(img, 'WORKLOAD GEN #', (x_start + 40, y_start - 35), self.FONT, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        # cv2.putText(img, 'LOAD', (x_start + 20, y_start - 150), self.FONT, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(img, 'GEN #', (x_start + 20, y_start - 130), self.FONT, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        
        cv2.putText(img, 'LOAD', (x_start + 20, y_start - 150), self.FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, 'GEN#', (x_start + 20, y_start - 130), self.FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        load_height = y_start - (y_start - 300)
        mid = load_height//2
        
        y_mid = y_start - mid
        
        spacing = (load_height-50)//10
        
        line_start_y = y_mid - (chain_id - 3)*spacing

        cv2.line(img, (x_start + 75, line_start_y), (x_start + 270, self.HEIGHT- 50 - 100 * (chain_id - 1)), (0, 0, 0), 2)

    def __display_load(self, img, load_history, nb_chain):
        # First display a x,y axis with the load function based on the history 
        fig, ax = plt.subplots()
        ax.clear()
        sns.lineplot(data = load_history, ax = ax)
        ax.set(xlabel = 't', ylabel = 'load')
        ax.set_xticks(range(0, 101, 10)) #len(load_history)))
        ax.set_yticks(range(0, 601, 100)) #len(load_history)))
        # ax.set_xticklabels(["" for _ in range(101)])
        # ax.set_xticklabels(["" for _ in range(101)])
        
        # from https://stackoverflow.com/questions/65457097/handling-matplotlib-figure-figure-using-opencv
        # figure = plt.gcf()
        # figure.canvas.draw()
        # b = figure.axes[0].get_window_extent()
        # img_sns = np.array(figure.canvas.buffer_rgba())
        # img_sns = img_sns[int(b.y0):int(b.y1),int(b.x0):int(b.x1),:]
        # img2 = cv2.cvtColor(img_sns, cv2.COLOR_RGBA2BGRA)
        #cv2.imshow('OpenCV',img2)
        #plt.savefig("test.jpg")
    
        # x = 50
        # y = self.HEIGHT - 400
        # __draw_x_y_axis(img, x, y)
        # print("load =", load)

        # Now display the load on each chain
        if len(load_history) == 0:
            return 
        last_load = int(load_history.to_numpy()[-1])
        for c in range(nb_chain):
            cv2.putText(img,
                        '%d' % (int(last_load / nb_chain)),
                        (250, self.HEIGHT - 40 - 100 * c),
                        self.FONT,
                        1,
                        (100, 100, 100),
                        2,
                        cv2.LINE_AA
                        )


    def __draw_action_history(self, img, actions_history):
        # x axis +arrow for time t, with t legend for action history
        x = 20
        y = self.HEIGHT - 40
        cv2.line(img, (x, y), (x + 150, y), (255, 0, 0), 3)
        cv2.line(img, (x + 140, y + 10), (x + 150, y), (255, 0, 0), 3)
        cv2.line(img, (x + 140, y - 10), (x + 150, y), (255, 0, 0), 3)
        cv2.putText(img, 't', (x + 160, self.HEIGHT - 30), self.FONT, 1, (0,0,0), 2, cv2.LINE_AA)
        
        # print("actions_history", actions_history)
        if len(actions_history) > 13:
            actions_history.pop(0)
            
        for i in range(len(actions_history)):
            # a = actions_history.to_numpy('int64')
            # print('a =', a)
            # a = a[i]
            a = actions_history[i]
            if a != 0:
                # cv2.line(img, (x + 10 + i * 10, y +  a * 10), (x + 10 + i * 10, y), (255, 0, 0), 3)
                # cv2.line(img, (x + 15 + i * 10, y +  a * 10), (x + 15 + i * 10, y), (255, 0, 0), 3)
                if a == 1:
                    c = (0, 0, 255)
                    v = -1
                else:
                    c = (0, 255, 0)
                    v = 1
                    # if action == 2, means we are scaling in (i.e. removing a VM), so  we set v to -1 to have it going in the other direction than when the platform is scaling out (where v == a == 1)
                    # v = a
                    # if v == 2:
                    # if a == 2:
                    #     # v = -1
                    #     v = 1
                    cv2.rectangle(img,(x + 10 + i * 10, y), (x + 15 + i * 10, y + v * 20), c, -1)
            else:
                # when no action is taken simply draw a small rectangle in the middle
                cv2.rectangle(img,(x + 10 + i * 10, y - 10), (x + 15 + i * 10, y + 10), (255, 0, 0), -1)
                pass

    def __draw_x_y_axis(self, img, x, y, xlen = 200, ylen = 200):
        # draw x and y axis (x,y) are (0,0)
        cv2.line(img, (x, y), (x + xlen, y), (0, 0, 0), 3)
        # cv2.putText(img, 'reward_x,y', (x,y), self.FONT, 1, (0,0,0), 2, cv2.LINE_AA)
        cv2.line(img, (x + xlen - 5, y - 5), (x + xlen, y), (0, 0, 0), 3)
        cv2.line(img, (x + xlen - 5, y + 5), (x + xlen, y), (0, 0, 0), 3)
        cv2.line(img, (x, y), (x, y - ylen), (0, 0, 0), 3)
        # cv2.line(img, (x, y), (x, y - 300), (0, 0, 0), 3) #SV for reward y-axis
        # cv2.line(img, (x, y), (x, y + ylen), (0, 0, 0), 3) #SV for reward
        cv2.line(img, (x - 5, y - ylen + 5), (x, y - ylen), (0, 0, 0), 3)
        cv2.line(img, (x + 5, y - ylen + 5), (x, y - ylen), (0, 0, 0), 3)
        
    # def __draw_reward(img, actions_reward_history):
    def __draw_reward(self, img, reward_arr):
        x = 875
        y = 220
        xlen=150
        ylen=200
        y_arw = 140
        self.__draw_x_y_axis(img, x, y, ylen=160)
        cv2.line(img, (x, y), (x, y + y_arw), (0, 0, 0), 3)
        
        cv2.line(img, (x - 5, y + y_arw - 5), (x, y + y_arw), (0, 0, 0), 3)
        cv2.line(img, (x + 5, y + y_arw - 5), (x, y + y_arw), (0, 0, 0), 3)
        
        cv2.line(img, (x-5, y+120), (x+5, y + 120), (0, 0, 0), 1) #For ylabel
        cv2.putText(img, "-1", (x+7, y+120), self.FONT, 0.5, (0,0,0), 1, cv2.LINE_AA)
        
        cv2.line(img, (x-5, y-130), (x+5, y - 130), (0, 0, 0), 1) #For xlabel
        cv2.putText(img, "1", (x+7, y-130), self.FONT, 0.5, (0,0,0), 1, cv2.LINE_AA)
        
        cv2.putText(img, "Reward", (x + 210, y), self.FONT, 1, (0,0,0), 2, cv2.LINE_AA)
        
        if len(reward_arr) > xlen:
            reward_arr.pop(0)
    
        for i in range(len(reward_arr)-1):
            # cv2.line(img, (x + xlen + i, int(y+ylen) - reward_arr[i]),
            #             (x + xlen + i + 1, int(y+ylen) - reward_arr[i + 1]), (255,0,0), 1)

            cv2.line(img, (x + i, int(y) - reward_arr[i]),
                     (x + i + 1, int(y) - reward_arr[i + 1]), (255,0,0), 1)


    def __draw_load_evolution(self, img, load_arr, total_ue):
        x = 875
        y = 650
        xlen=150
        ylen=200

        self.__draw_x_y_axis(img, x, y)
        cv2.putText(img, "Load", (x + 210, y), self.FONT, 1, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, f"Total UEs: {total_ue}", (x + 50, y-250), self.FONT, 1, (0,0,0), 1, cv2.LINE_AA)
        
        cv2.line(img, (x-5, y-170), (x+5, y - 170), (0, 0, 0), 1) #For xlabel
        cv2.putText(img, "1250", (x+7, y-170), self.FONT, 0.5, (0,0,0), 1, cv2.LINE_AA)
        
        cv2.putText(img, "0", (x-2, y+20), self.FONT, 0.5, (0,0,0), 1, cv2.LINE_AA)
        
        if len(load_arr) > xlen:
            load_arr.pop(0)
            
        for i in range(len(load_arr)-1):
            # cv2.line(img, (x + xlen + i, int(y+ylen) - reward_arr[i]),
            #             (x + xlen + i + 1, int(y+ylen) - reward_arr[i + 1]), (255,0,0), 1)
            
            cv2.line(img, (x + i, int(y) - load_arr[i]),
                     (x + i + 1, int(y) - load_arr[i + 1]), (255,0,0), 1)
            
    def __insert_agent(self, img, data):
        
        action = data['action'].to_numpy(dtype='int64')[0] #SV
        
        action_map = {0: 'NO ACTION', 1: 'SCALE UP', 2: 'SCALE DOWN'}
        
        x_offset=30
        y_offset=60
        
        img[y_offset:y_offset + self.AGENT_LOGO.shape[0], x_offset:x_offset + self.AGENT_LOGO.shape[1]] = self.AGENT_LOGO
        
        t_pos_x = x_offset + self.AGENT_LOGO.shape[0] + 30
        t_pos_y = y_offset + (self.AGENT_LOGO.shape[1]//2+10)
        
        cv2.putText(img, f"{action_map.get(action)}", (t_pos_x, t_pos_y), self.FONT, 0.9, (0,0,255), 2, cv2.LINE_AA)
        
        # cv2.putText(img, "D3QN AGENT", (x_offset-AGENT_LOGO.shape[0]//4, y_offset+AGENT_LOGO.shape[1]+20), self.FONT, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, "D3QN", (x_offset+20, y_offset + self.AGENT_LOGO.shape[1]+20), self.FONT, 0.6, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(img, "AGENT", (x_offset+20, y_offset + self.AGENT_LOGO.shape[1]+40), self.FONT, 0.6, (0,0,0), 1, cv2.LINE_AA)


    def __gen_frame(self, t):
        print("t =",t)
        img = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)
        img.fill(255) # white background
        
        # vertical separator between platform and metrics
        cv2.line(img, (855, 0), (855, self.HEIGHT), (0, 0, 0), 3)
        
        # t= timestamp in top left corner
        # cv2.putText(img, 't='+str(t), (50, 30), self.FONT, 1, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, 'timestep = '+str(t), (350, 50), self.FONT, 1, (0,0,0), 2, cv2.LINE_AA)
        
        
        # df = __generate_pseudo_data(DURATION) #Omit out the generation of pseudo data
        # data = df.loc[[df.index[t]]]
        # history_t0 = t - 10
        # if history_t0 < 0:
        #     history_t0 = 0
        #actions_history = df[(df.index >= history_t0) & (df.index < t)]['action']
        data, actions_history, load_history, state_logs = self.__get_last_state(history_len = 10)
        #data = df.loc[[df.index[-1]]]
        
        #SV to check performance over the dataframe
        data = data.loc[[data.index[t]]]
        
        nb_chain = data['vm_count'].to_numpy(dtype='int64')[0] #SV
        # nb_chain = int(data['vm_count'])
        grouped_vm = self.__get_seg_data(state_logs)
        
        # data = df.loc[[df.index[t]]]
        # n = data['nbVM']
        # nb_chain = n.to_numpy()[0]
        for c in range(nb_chain) :
            self.__draw_chain(img, c + 1, t, grouped_vm)

        # print("actions_history\n", actions_history, "\n")
        action = data['action'].to_numpy(dtype='int64')[0] #SV
        self.ACTION_PAST_VALS.append(action)
        # __draw_action_history(img, actions_history)
        self.__draw_action_history(img, self.ACTION_PAST_VALS)
        
        
        # load_t0 = t - 100
        # if load_t0 < 0:
        #     load_t0 = 0
        # load_history = df[(df.index >= load_t0) & (df.index < t)]['load']
        # __display_load(img, load_history, nb_chain)
        reward_val = data['reward'].to_numpy(dtype='float64')[0] #SV
        reward_val = int(reward_val*110) #SV
        # reward_val = int(data['reward'] * 100)
        self.REWARD_PAST_VALS.append(reward_val)
        
        #__draw_reward(img, [])
        self.__draw_reward(img, self.REWARD_PAST_VALS)
        
        ue_connected_agg = data['ue_connected'].to_numpy(dtype='float64')[0] #SV
        # ue_connected_agg = int(data['ue_connected'])
        # ue_connected_agg = int(ue_connected_agg*100) #SV
        # ue_connected_agg = int(ue_connected_agg*100) #SV
        total_ue = self.__get_total_ue(grouped_vm,t,'ue_connected')
        
        # LOAD_PAST_VALS.append(int(ue_connected_agg*100))
        self.LOAD_PAST_VALS.append(int(total_ue // 5))

    

        self.__draw_load_evolution(img, self.LOAD_PAST_VALS, total_ue)

        # img.paste(AGENT_LOGO, (50, 100), mask=AGENT_LOGO)
        self.__insert_agent(img, data)

    
        
        return img

    # @webapp.route('/')
    # def webpage():
    #    return render_template('render.html')

    def render_env(self, env):
        # host_name = "0.0.0.0"
        # port = 8088
        # threading.Thread(target=lambda: webapp.run(host = host_name,
        #                                            port = port,
        #                                            debug = True,
        #                                            use_reloader = False)).start()
    
        # print(df.to_string())
        
        # for t in range(DURATION):
        t = env.step_idx
        img = self.__gen_frame(t)
        # filtered_image = cv2.GaussianBlur(img, (5, 5), 0)
        # # img_array.append(img)
        # img_array.append(filtered_image)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        
        # Sharpen the image
        sharpened_image = cv2.filter2D(img, -1, kernel)
        self.img_array.append(sharpened_image)
        if self.WRITE_IMGS:
            name = self.IMG_FOLDER + 'img_' + str(t) + ".jpg"
            cv2.imwrite(name, img)
            
            # frame = img.tobytes()
            #frame = cv2.imread(name).tobytes()
            # print(img)
            # print(frame)
            #yield (b'--frame\r\n'
            #       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            
            #t += 1
            #    time.sleep(1)
            
    def save_video(env):
        # now save the video
        framerate = 1
        # framerate=30
        size = (self.WIDTH, self.HEIGHT)
        out_filename = 'opencv_plots_workload.mp4'
        out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'mp4v'), framerate, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
            out.release()

"""
Main 
"""
if __name__ == '__main__':
    __gen_frames()
