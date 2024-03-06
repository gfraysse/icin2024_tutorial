import os
import sys
import logging
import wandb
from utilities import constants as CONSTANTS
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt

class IOHelper:
    def __init__(self, writer_dir):
        substr = dt.now().strftime('%Y-%m-%d-%H-%M')

        wandb.init(project= CONSTANTS.PROJECT_NAME,
                    entity=None,
                    sync_tensorboard=True,
                    name=f"{CONSTANTS.PROJECT_NAME}_{substr}",
                    config=CONSTANTS.MODEL_CONFIGS)

        self.writer = SummaryWriter(f"{writer_dir}/pre_mainNetwork_{substr}")

    def get_logger():
        logger = logging.getLogger(__name__)

        logger.setLevel(logging.DEBUG)
        format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(format)
        logger.addHandler(ch)

        return logger
    
    def check_dirs(*directories):
        for dir in directories:
            if not os.path.isdir(dir):
                 os.mkdir(dir)

    def write_target_network_update(self, target_update_steps, targetNetwork_update, global_step):
        self.writer.add_scalar(f'RL_parameters/targetNetwork_update_{target_update_steps}', targetNetwork_update, global_step)
    
    def __update_env_tbd(self, env_instance, info, max_limits, total_time_steps):
        vm_count = env_instance.unwrapped.num_instances
        self.writer.add_scalar("env_info/vm_count",vm_count , total_time_steps)

        next_state = env_instance.unwrapped.next_state
        
        for k,v in next_state.items():
            if k == 'vm_count':
                continue
            elif k == 'mem_free':
                value = v * 4 
                # Since we want unnormalized value in GBs
                # v * (4*1024) to get unnormalized in MBs and (v * 4*1024) / 1024 ~ v * 4 
                self.writer.add_scalar(f"env_info/avg_{k}", value, total_time_steps)

            elif k in ['ue_connected','num_calls_dropped']:
                value = v * max_limits[k]
                self.writer.add_scalar(f"env_info/aggregated_{k}", value, total_time_steps)
            else:
                value = v * max_limits[k]
                self.writer.add_scalar(f"env_info/avg_{k}", value, total_time_steps)

        self.writer.add_scalar("env_info/crash_count", info['nb_crash'] , total_time_steps)

        self.writer.add_scalar(f"env_info/normalized_num_calls_dropped", next_state['num_calls_dropped'], total_time_steps)

    def __update_state_info_tbd(self, env_instance, total_time_steps):

        next_state_dict = env_instance.unwrapped.next_state
       
        self.writer.add_scalar(f"next_state/ue_connected", next_state_dict['ue_connected'], total_time_steps)
        self.writer.add_scalar(f"next_state/ue_attach_rate",  next_state_dict['ue_attach_rate'], total_time_steps)
        self.writer.add_scalar(f"next_state/cpu_load", next_state_dict['cpu_percentage'], total_time_steps)
        self.writer.add_scalar(f"next_state/mem_free", next_state_dict['mem_free'], total_time_steps)
        self.writer.add_scalar(f"next_state/mme_service_memory_usage", next_state_dict['magma_mme_service'], total_time_steps)
        self.writer.add_scalar(f"next_state/vm_count", next_state_dict['vm_count'], total_time_steps)


    def __log_tbd_instance_info(self, env_instance, total_time_steps):
        """Method to log metrics of each Instance on Tensorboard"""

        instances = env_instance.unwrapped.instances

        for instance in instances:

            self.writer.add_scalar(f"VM_info/VM_{instance.get_id()}_ue_connected", instance.get_connected_users(), total_time_steps)
            self.writer.add_scalar(f"VM_info/VM_{instance.get_id()}_ue_attach_rate", instance.get_attach_rate(), total_time_steps)
            self.writer.add_scalar(f"VM_info/VM_{instance.get_id()}_mem_free", instance.get_mem_free(), total_time_steps)
            self.writer.add_scalar(f"VM_info/VM_{instance.get_id()}_magma_mme_service", instance.get_mme_usage(), total_time_steps)
            self.writer.add_scalar(f"VM_info/VM_{instance.get_id()}_num_calls_dropped", instance.get_nb_calls_dropped(), total_time_steps)

    def update_board(self, env, info, max_limits, total_time_steps):
        self.__update_env_tbd(env, info, max_limits, total_time_steps)
        self.__log_tbd_instance_info(env, total_time_steps)
        self.__update_state_info_tbd(env, total_time_steps)




    def create_network_graphs(self, d3qn_agent):
        self.writer.add_graph(d3qn_agent.onlineQNetwork, d3qn_agent.batch_state)

    def write_loss(self, loss, epsilon, beta, learn_steps):
        self.writer.add_scalar('RL_parameters/loss', loss.item(), global_step=learn_steps)
        self.writer.add_scalar('RL_parameters/epsilon', epsilon, global_step=learn_steps)
        self.writer.add_scalar('RL_parameters/beta', beta, global_step=learn_steps)

    def write_buffer_size(self, buffer_size, total_time_steps):
        self.writer.add_scalar('RL_parameters/buffer_size', buffer_size, global_step=total_time_steps)

    def write_summary(self, d3qn_agent, time_steps, epoch, episode_reward, total_time_steps):
        self.writer.add_scalar('iterations/steps_per_epoch', time_steps ,  global_step=epoch)
        self.writer.add_scalar('RL_parameters/episode_reward', episode_reward, global_step=epoch)
        self.writer.add_scalar('iterations/epoch_steps', epoch, global_step=total_time_steps)
        self.writer.add_histogram('mainQAdv', d3qn_agent.onlineQNetwork.adv.weight)
        self.writer.add_histogram('mainQValue', d3qn_agent.onlineQNetwork.value.weight)
        self.writer.add_histogram('TargetAdv', d3qn_agent.onlineQNetwork.adv.weight)
        self.writer.add_histogram('TargetValue', d3qn_agent.onlineQNetwork.value.weight)

    def flush_writers(self):
        self.writer.flush()