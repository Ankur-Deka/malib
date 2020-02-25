import json
import os, time
from multiprocessing import Process

from malib.agents.agent_factory import *
from malib.environments import DifferentialGame
from malib.environments.particle import make_particle_env
from malib.logger.utils import set_logger
from malib.samplers.sampler import SingleSampler, MASampler
from malib.trainers import SATrainer, MATrainer
from malib.utils.random import set_seed
import numpy as np, pickle

def svd_sol(A, b):
        U, sigma, Vt = np.linalg.svd(A)
        sigma[sigma<1e-10] = 0
        sigma_reci = [(1/s if s!=0 else 0) for s in sigma]
        sigma_reci = np.diag(sigma_reci)
        x = Vt.transpose().dot(sigma_reci).dot(U.transpose()).dot(b)
        return(x)

def isCaught(observation_n, th):
    pos = np.array([obs[2:4] for obs in observation_n])
    posAdv, posAgent = pos[:3,:], pos[3,:]
    dists = np.sqrt(np.sum(np.square(posAdv-posAgent), axis = 1))
    
    caught = False
    if np.max(dists) <= th:
        A = np.concatenate((posAdv.transpose(), np.ones((1,3))), axis = 0)
        b = np.concatenate((posAgent, np.ones(1))).reshape(3,1)
        alpha = svd_sol(A,b)
        if all(alpha>=0) and all(alpha<=1):
            caught = True
    return(caught)

def extract_vel_pos(current_observation_n):
    return(np.array([obs[:4] for obs in current_observation_n]))

def main():
    #  PR2 - empirical estimation of opponent conditional policy
    #  PR2S - soft estimation of opponent conditional policy
    settings = [
        # 'ROMMEO_ROMMEO_ROMMEO_ROMMEO',
        # 'PR2S_PR2S_PR2S',
        # 'PR2_PR2_PR2',
        # 'PR2_PR2',
        # 'PR2_PR2_PR2_PR2',
        'PR2_PR2_PR2_PR2',
        # 'PR2_PR2_PR2_DDPG'
        # 'SAC_SAC_SAC',
        # 'DDPG_DDPG_DDPG_DDPG',
    ]
    # game = 'simple_spread'
    # game = 'simple_adversary'
    # game = 'simple_push'
    # game = 'simple_tag'
    game = 'simple_predator_prey'
    # game = 'fortattack-v0'

    game_name = 'simple_predator_prey_diff_and_collision_rew' 
    # game_name = None
    if game_name is None:
        game_name = 'simple_predator_prey_diff_and_collision_rew'

    checkpoint = 120000
    num_episodes = 20   # valid only for npy file format, csv deals with a single episode

    out_file = 'out_files/'+game_name
    file_format = 'npy' # 'csv' or 'npy'

    if file_format == 'csv':
        num_episodes = 1

    all_obs = {}

    for setting in settings:
        seed = 1 + int(23122134 / (3 + 1))
        path = 'saved_agents/'+game_name+'/'+setting+'/agents_ckpt_'+str(checkpoint-1)+'.pickle'
        # path = 'saved_agents/'+game+'/'+setting+'_individual_reward_only/agents_ckpt_'+str(checkpoint-1)+'.pickle'
        # path = 'saved_agents/'+game+'_different_rew/'+setting+'/agents_ckpt_'+str(checkpoint-1)+'.pickle'
        with open(path, 'rb') as f:
            agents = pickle.load(f)


    env = make_particle_env(game)
    gymAgents = env.world.agents
    
    th = 0.4       # 0.3
    caughtCount = 0
    totalCaughtSteps = 0

    for i in range(num_episodes):
        print(i)
        current_observation_n = env.reset()
        
        eps_obs = []
        for j in range(200):
            action_n = []
            # print(type(current_observation_n), type(current_observation_n[0]), current_observation_n[0].shape, np.vstack(current_observation_n).shape)
            # print([obs.shape for obs in current_observation_n])
            # print(extract_vel_pos(current_observation_n))
            eps_obs.append(extract_vel_pos(current_observation_n))
            for agent, current_observation in zip(agents, current_observation_n):
                    action = agent.act(current_observation.astype(np.float32))
                    action_n.append(np.array(action))
            next_observation_n, reward_n, done_n, info = env.step(action_n)
            

            # pos = np.array([obs[2:4] for obs in next_observation_n])
            # dists = np.sqrt(np.sum(np.square(pos[:3,:]-pos[3,:]), axis = 1))
            # print(dists)
            # print('NEW ',np.array([obs[2:4] for obs in next_observation_n]))
            
            env.render(mode="rgb_array")[0]
            time.sleep(0.03)
            # if np.max(dists) <= th:
            if isCaught(next_observation_n, th):
                caughtCount += 1
                totalCaughtSteps += j+1
                break
            current_observation_n = next_observation_n
        time.sleep(2)

        if not out_file is None:
            all_obs[str(i)] = np.array(eps_obs)
    
    
    
    if file_format == 'csv':
        obs = np.vstack(all_obs[npy])
        np.savetxt(out_file, obs, delimiter=",")

    elif file_format == 'npy':
        print(type(all_obs))
        np.save(out_file, all_obs)
    
    print('Caught {}/{} times'.format(caughtCount, num_episodes))

    if caughtCount:
        print('Average time steps to catch {}'.format(totalCaughtSteps/caughtCount))
if __name__ == '__main__':
    main()
