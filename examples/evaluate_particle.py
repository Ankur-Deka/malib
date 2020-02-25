# Created by yingwen at 2019-03-16

import json
import os
from multiprocessing import Process

from malib.agents.agent_factory import *
from malib.environments import DifferentialGame
from malib.environments.particle import make_particle_env
from malib.logger.utils import set_logger
from malib.samplers.sampler import SingleSampler, MASampler
# from malib.trainers import SATrainer, MATrainer
from malib.evaluators import MAEvaluator
import malib.evaluators
from malib.utils.random import set_seed
import numpy as np, pickle


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


def test_fixed(seed, agent_setting, game_name='ma_softq'):
    set_seed(seed)
    suffix = f'fixed_play/{game_name}/{agent_setting}/{seed}'

    set_logger(suffix)      # This is the function which is initializing the log files

    batch_size = 1 ##1024
    training_steps = 25*60000       ## 25 steps in MAtrainer = 1 episode
    exploration_steps = 2000
    save_after = 25*500             ## save after every save_after training steps
    max_replay_buffer_size = 1e5
    hidden_layer_sizes = (100, 100)
    max_path_length = 25
    save_path = 'saved_agents/'+game_name+'/'+agent_setting
    ckpt = 30000-1
    render_after = 1

    agent_num = 4
    env = make_particle_env(game_name)
    agents = []
    agent_types = agent_setting.split('_')
    assert len(agent_types) == agent_num
    for i, agent_type in enumerate(agent_types):
        agents.append(get_agent_by_type(agent_type, i, env, hidden_layer_sizes=hidden_layer_sizes,
                                        max_replay_buffer_size=max_replay_buffer_size))

    sampler = MASampler(agent_num, batch_size=batch_size, max_path_length=max_path_length, render_after=render_after)

    sampler.initialize(env, agents)

    evaluator = MAEvaluator(env=env, agents=agents, sampler=sampler, ckpt = ckpt, steps=training_steps,
                        extra_experiences=['annealing', 'recent_experiences'],
                        max_path_length=max_path_length,
                        batch_size=batch_size,
                        save_path = save_path)
    # sampler.initialize(env, evaluator.agents)
    # evaluator.restore(restore_path)
    evaluator.demo()    ## run single episode of game with rendering
    
def main():
    #  PR2 - empirical estimation of opponent conditional policy
    #  PR2S - soft estimation of opponent conditional policy
    settings = [
        # 'ROMMEO_ROMMEO_ROMMEO',
        # 'PR2S_PR2S_PR2S',
        'PR2_PR2_PR2_PR2',
        # 'SAC_SAC_SAC',
        # 'DDPG_DDPG_DDPG',
    ]
    game = 'simple_predator_prey'
    for setting in settings:
        seed = 1 + int(23122134 / (3 + 1))
        test_fixed(seed, setting, game)

if __name__ == '__main__':
    main()
