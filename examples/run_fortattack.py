# Created by yingwen at 2019-03-16

import json
import os
from multiprocessing import Process

from malib.agents.agent_factory import *
from malib.environments import DifferentialGame
from malib.environments.fortattack import make_fortattack_env
from malib.logger.utils import set_logger
from malib.samplers.sampler import SingleSampler, MASampler
from malib.trainers import SATrainer, MATrainer
from malib.utils.random import set_seed
import numpy as np, pickle
import gym, gym_fortattack

def get_agent_by_type(type_name, i, env,
                      hidden_layer_sizes,
                      max_replay_buffer_size):
    if type_name == 'SAC':
        return get_sac_agent(env, hidden_layer_sizes=hidden_layer_sizes,
                             max_replay_buffer_size=max_replay_buffer_size, policy_type='gumble')
    elif type_name == 'PR2':
        return get_pr2_agent(env, agent_id=i,
                                hidden_layer_sizes=hidden_layer_sizes,
                                max_replay_buffer_size=max_replay_buffer_size, policy_type='gumble')
    elif type_name == 'PR2S':
        return get_pr2_soft_agent(env, agent_id=i,
                                hidden_layer_sizes=hidden_layer_sizes,
                                max_replay_buffer_size=max_replay_buffer_size, policy_type='gumble')
    elif type_name == 'ROMMEO':
        return get_rommeo_agent(env, agent_id=i,
                                hidden_layer_sizes=hidden_layer_sizes,
                                max_replay_buffer_size=max_replay_buffer_size, policy_type='gumble')
    elif type_name == 'DDPG':
        return get_ddpg_agent(env, agent_id=i,
                              hidden_layer_sizes=hidden_layer_sizes,
                              max_replay_buffer_size=max_replay_buffer_size, policy_type='gumble')

def train_fixed(seed, agent_setting, game_name, cont_ckpt = 0):
    set_seed(seed)
    suffix = f'fixed_play/{game_name}/{agent_setting}/{seed}'
    # old_time = '20191008-112304-cont'
    # set_logger(suffix, old_time = old_time)
    set_logger(suffix)

    batch_size = 1024
    max_path_length = 100
    training_steps = max_path_length*60000		## 25 steps in MAtrainer = 1 episode
    exploration_steps = max_path_length*80
    save_after = max_path_length*1000				## save after every save_after training steps
    max_replay_buffer_size = 1e5
    hidden_layer_sizes = (100, 100)
    render_after = None#1
    save_path = 'saved_agents/'+game_name+'/'+agent_setting
    # save_path = 'saved_agents/'+game_name+'/'+agent_setting

    agent_num = 4
    env = make_fortattack_env()
    agents = []
    agent_types = agent_setting.split('_')
    ## here, agents means the algorithms, inside the environment agents means the world agents
    assert len(agent_types) == agent_num
    for i, agent_type in enumerate(agent_types):
        agents.append(get_agent_by_type(agent_type, i, env, hidden_layer_sizes=hidden_layer_sizes,
                                        max_replay_buffer_size=max_replay_buffer_size))

    # for agent in agents:
    #     with open('save_path', 'wb') as f:
    #         pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)
    sampler = MASampler(agent_num, batch_size=batch_size, max_path_length=max_path_length, render_after=render_after)
    sampler.initialize(env, agents)


    trainer = MATrainer(env=env, agents=agents, sampler=sampler, steps=training_steps,
                        max_path_length = max_path_length,
                        exploration_steps=exploration_steps,
                        training_interval=10,
                        extra_experiences=['annealing', 'recent_experiences'],
                        batch_size=batch_size,
                        save_path = save_path,
                        save_after = save_after,
                        cont_ckpt = cont_ckpt)
    
    # Either continue the training or resume training
    if cont_ckpt == 0:
        trainer.run()
    else:
        trainer.resume()
    
def main():
    #  PR2 - empirical estimation of opponent conditional policy
    #  PR2S - soft estimation of opponent conditional policy
    settings = [
        #'ROMMEO_ROMMEO_ROMMEO_ROMMEO',
        #'ROMMEO_ROMMEO_ROMMEO_DDPG',
        #'DDPG_DDPG_DDPG_ROMMEO',
        # 'PR2S_PR2S_PR2S',
        # 'PR2_PR2_PR2',
        # 'PR2_PR2',
        'PR2_PR2_PR2_PR2',
    ]
    game_name = 'fortattack-v0'
    cont_ckpt = 60000

    for setting in settings:
        seed = 1 + int(23122134 / (3 + 1))
        train_fixed(seed, setting, game_name, cont_ckpt = cont_ckpt)

if __name__ == '__main__':
    main()
