import logging
import os
import os.path
import numpy as np
import torch
import gymnasium as gym
import argparse

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
    global TARGET_UPDATE_STEPS, EPOCHS, MIN_BUFFER_SIZE_FOR_TRAIN, STEPS, EXPLORE
    parser = argparse.ArgumentParser(
        prog='ICIN tutorial',
        description='ICIN tutorial',
        epilog=''
    )
    parser.add_argument('-E', '--env', action = 'store', type = str, required = False)
    parser.add_argument('-s', '--nb_steps', action = 'store', type = int, required = False)
    parser.add_argument('-e', '--nb_episodes', action = 'store', type = int, required = False)
    parser.add_argument('-b', '--buffer_size', action = 'store', type = int, required = False)
    parser.add_argument('-x', '--exploration_duration', action = 'store', type = int, required = False)
    parser.add_argument('-t', '--target_update_steps', action = 'store', type = int, required = False)
    args = parser.parse_args(sys.argv[1:])
    
    #Check for directories to store metadata
    IOHelper.check_dirs(BACKUPS_PATH, 
                        SAVED_MODELS_PATH, 
                        SAVED_BUFFER_PATH,
                        WORKLOAD_LOGS_PATH,
                        TENSORBOARD_ROOT_DIR,
                        TENSORBOARD_RUNS_DIR)

    logger = IOHelper.get_logger()    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.env:
        print("env", args.env)
        if args.env == "TelcoCoreScaling":
            env = gym.make("telco_core_scaling.envs:TelcoCoreScaling-v0")
        elif args.env == "CartPole":
            env = gym.make('CartPole-v1', render_mode = 'human')
        elif args.env == "Acrobot":
            env = gym.make('Acrobot-v1', render_mode = 'human')
        elif args.env == "Pendulum":
            env = gym.make('Pendulum-v1', render_mode = 'human')
        elif args.env == "MountainCar":
            env = gym.make('MountainCar-v0', render_mode = 'human')
        elif args.env == "LunarLander":
            env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0,
                           enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode = 'human')
        else:
            print("Unknown env: ", args.env)
            sys.exit(1)
    else:
        # TUTORIAL: select the environment you want to use
        # env = gym.make(
        #     "telco_core_scaling.envs:TelcoCoreScaling-v0"
        # )
        env = gym.make('CartPole-v1', render_mode = 'human')
        # env = gym.make('Acrobot-v1', render_mode = 'human')
        # env = gym.make('Pendulum-v1', render_mode = 'human')
        # env = gym.make('MountainCar-v0', render_mode = 'human')
        # env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0,
        #                enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode = 'human')

    if args.nb_episodes:
        print("nb_episodes", args.nb_episodes)
        EPOCHS = args.nb_episodes
    if args.nb_steps:
        print("nb_steps", args.nb_steps)
        STEPS = args.nb_steps
    if args.buffer_size:
        print("buffer_size", args.buffer_size)
        MIN_BUFFER_SIZE_FOR_TRAIN = args.buffer_size
    if args.exploration_duration:
        print("exploration_duration", args.exploration_duration)
        EXPLORE = args.exploration_duration
    if args.target_update_steps:
        print("target_update_steps", args.target_update_steps)
        TARGET_UPDATE_STEPS = args.target_update_steps

    try:
        num_metrics = len(METRICS_LIST)
        num_actions = len(env.get_action_space())
    except Exception as e:
        num_actions = env.action_space.n 
        num_metrics = env.observation_space.shape[0]
        env.reset()
        
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

    is_done = False
    try:
        for epoch in range(EPOCHS):
            logger.info("\n\n" + "=" * 40 + "Epoch ->" + str(epoch) + "=" * 40)
            state, _ = env.reset()
            episode_reward = 0
            for time_steps in range(STEPS):
                
                logger.info("\n\n" + "-" * 40 + str(time_steps) + "-" * 40)
                action = d3qn_agent.get_action(state=state, epsilon=epsilon, seed=ACTION_SEED)
                
                next_state, reward, truncated, done, info = env.step(action)
                total_time_steps += 1
                episode_reward += reward
                if "TelcoCoreScaling" in str(env):
                    max_limits = env.get_limits()
                    
                    ## env info
                    writer_helper.update_board(env, info, max_limits, total_time_steps)
                else:                    
                    env.render()
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
                if "TelcoCoreScaling" not in str(env) and truncated:
                    break
                if done:
                    is_done = True
                    break
                state = next_state

                d3qn_agent.save_memory_replay()
                
                if total_time_steps % 50 == 0:
                    d3qn_agent.save_model(epoch, SAVED_MODELS_PATH)
                    
            writer_helper.write_summary(d3qn_agent, time_steps, epoch, episode_reward, total_time_steps)
            if "TelcoCoreScaling" not in str(env) and is_done:
                break
        
    except KeyboardInterrupt:
        writer_helper.flush_writers()
        try:
            env.shutdown()
        except:
            pass

    finally:
        pass
    logger.info("Pipeline Run Finished")
    writer_helper.flush_writers()

if __name__ == "__main__":
    main()
