import logging
import os
import os.path
import numpy as np
import torch
import gymnasium as gym

from network_architecture.D3QNAgent import D3QNAgent

from utilities.constants import *
from utilities.io_helper import IOHelper

def resume_training(d3qn_agent, model):
    saved_models = sorted(os.listdir(SAVED_MODELS_PATH))
    if os.path.isdir(os.path.abspath(os.path.join(SAVED_MODELS_PATH, saved_models[-1]))):
        saved_models.pop(-1)

    resumed_model = saved_models[-1]
    resumed_model = model or os.path.abspath(os.path.join(SAVED_MODELS_PATH, resumed_model))
    logging.info("\nResuming Training from model %s" % resumed_model)
    model_info_dict = torch.load(resumed_model)
    
    d3qn_agent.resume_training(model_info_dict)

def main():
    #Check for directories to store metadata
    IOHelper.check_dirs(BACKUPS_PATH, 
                        SAVED_MODELS_PATH, 
                        SAVED_BUFFER_PATH,
                        WORKLOAD_LOGS_PATH,
                        TENSORBOARD_ROOT_DIR,
                        TENSORBOARD_RUNS_DIR)

    logger = IOHelper.get_logger()    
    
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: Instatiate environment
    env = gym.make(
        "telco_core_scaling.envs:TelcoCoreScaling-v0"
    )

    num_metrics = len(METRICS_LIST)
    num_actions = len(env.get_action_space())

    logging.info(f"shape of input state : ( {num_metrics}, 1)")

    d3qn_agent = D3QNAgent(num_metrics=num_metrics, 
                           num_actions=num_actions, 
                           device=device, 
                           optimizer=OPTIM, 
                           initial_lr=INITIAL_LR,
                           buffer_size=REPLAY_MEMORY,
                           buffer_path=SAVED_BUFFER_PATH,
                           buffer_name=BUFFER_NAME,
                           per_alpha=PER_ALPHA,
                           seed=ACTION_SEED)

    model = 0 #"/home/cloud/rl_pipeline/src/saved_models/save_model_2022_10_27_17_34.pth" # "/home/cloud/rl_pipeline/src/saved_models/save_model_2022_10_06_05_58.pth"
    if os.path.isdir(SAVED_MODELS_PATH) and RESUME_TRAINING:
        resume_training(d3qn_agent, model)
    
    writer_helper = IOHelper(TENSORBOARD_RUNS_DIR)

    epsilon = INITIAL_EPSILON
    beta = INITIAL_BETA
    learn_steps = 0

    begin_learn = False
    total_time_steps = 0
    episode_reward = 0
    targetNetwork_update = 1
    writer_helper.write_target_network_update(TARGET_UPDATE_STEPS, targetNetwork_update, global_step=total_time_steps)

    try:
        for epoch in range(EPOCHS):
            logger.info("\n\n" + "=" * 40 + "Epoch ->" + str(epoch) + "=" * 40)
            state, _ = env.reset()
            episode_reward = 0
            for time_steps in range(STEPS):
                
                logger.info("\n\n" + "-" * 40 + str(time_steps) + "-" * 40)
                action = d3qn_agent.get_action(state=state, epsilon=epsilon, seed=ACTION_SEED)
                
                next_state, reward, _, done, info = env.step(action)
                max_limits = env.get_limits()
                total_time_steps += 1
                episode_reward += reward
                
                ## env info
                writer_helper.update_board(env, info, max_limits, total_time_steps)
        
                d3qn_agent.memory_replay.populate((state, next_state, action, reward, done))

                buffer_size = d3qn_agent.get_current_buffer_size()
                logger.info("\n\n" +"=" * 40 + "memory_replay size"  + str(buffer_size))
                
                

                if buffer_size >= MIN_BUFFER_SIZE_FOR_TRAIN:
                    if begin_learn is False:
                        logger.info('\n\n***Learn begin!***')
                        begin_learn = True
                    learn_steps += 1
        
                    beta = np.min([1., beta + BETA_INC])
                    d3qn_agent.forward(beta, BATCH, GAMMA)
                    
                    if epoch==0 and time_steps==0: 
                        writer_helper.create_network_graphs(d3qn_agent)
 
                    loss = d3qn_agent.calc_loss()
                    d3qn_agent.optim_step()

                    
                    logger.info("loss: {}".format(loss.item()))
                    
                    writer_helper.write_loss(loss=loss, epsilon=epsilon, beta=beta, learn_steps=learn_steps)
                    writer_helper.write_buffer_size(buffer_size=buffer_size, total_time_steps=total_time_steps)

                    if learn_steps % TARGET_UPDATE_STEPS == 0:
                        d3qn_agent.update_target_network()
                        targetNetwork_update+=1
                        writer_helper.write_target_network_update(TARGET_UPDATE_STEPS, targetNetwork_update, global_step=total_time_steps) 

                    if epsilon > FINAL_EPSILON:
                        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                
                writer_helper.write_buffer_size(buffer_size=buffer_size, total_time_steps=total_time_steps)

                logger.info('====Ep {}\tMoving average score: {:.2f}\tSteps: {}===='.format(epoch, episode_reward, time_steps))
                if done:
                    break
                state = next_state

                d3qn_agent.save_memory_replay()
                
                if total_time_steps % 50 == 0:
                    d3qn_agent.save_model(epoch, SAVED_MODELS_PATH)
                    
            writer_helper.write_summary(d3qn_agent, time_steps, epoch, episode_reward, total_time_steps)
        
    except KeyboardInterrupt:
        writer_helper.flush_writers()
        env.shutdown()

    finally:
        pass
    logger.info("Pipeline Run Finished")
    writer_helper.flush_writers()

if __name__ == "__main__":
    main()