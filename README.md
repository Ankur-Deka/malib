[![Build](https://travis-ci.com/ying-wen/malib.svg?branch=master)](./VERSION)
[![Supported TF Version](https://img.shields.io/badge/TensorFlow-2.0.0%2B-brightgreen.svg)](https://github.com/tensorflow/tensorflow/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

# Multi-Agent Reinforcement Learning Framework

In this fork, I've made a new environment called simple_predator_prey, by modifying the simple_tag multi agent particle environment. The file is ```multiagent/scenarios/simple_predator_prey.py```

The environment has the following features:
1) Predators catch the prey when all of them are close enough to the prey
2) Predators can be encouraged to surround the prey from all sides by setting the 

This Framework aims to provide an easy to use toolkit for
Multi-Agent Reinforcement Learning research.
Overall architecture:

![processes](./docs/architecture.png)

Environment: There are two differences for Multi-Agent Env Class: 1. The step(action_n) accepts n actions at each time; 2. The Env class needs a MAEnvSpec property which describes the action spaces and observation spaces for all agents.

Agent: the agent class has no difference than common RL agent, it uses the MAEnvSpec from Env Class to init the policy/value nets and replay buffer.

MASampler: Because the agents have to rollout simultaneously, a MASampler Class is designed to perform the sampling steps and add/return the step tuple to each agent's replay buffer.

MATrainer: In single agent, the trainer is included in the Agent Class. However, due to the complexity of Multi-Agent Training, which has to support independent/centralized/communication/opponent modelling, it is necessary to have a MATrainer Class to abstract these requirements from Agent Class. This is the core for Multi-agent training.

## Installation

Required Python Version: >= 3.6 

* Using Local Python Environment:

```shell
cd malib
conda env create --file=environment2.yml
conda activate malib
conda develop ./
```


## Train

```shell
cd examples
python run_particle.py
```

## Testing Code

```shell
python -m pytest tests
```

Testing With Keyword

```shell
python -m pytest tests -k "environments"
```

## Reference Projects
The project implementation has referred much and adopted some codes from the following projects: [agents](https://github.com/tensorflow/agents), [maddpg](https://github.com/openai/maddpg), [softlearning](https://github.com/rail-berkeley/softlearning), [garage](https://github.com/rlworkgroup/garage), [markov-game](https://github.com/aijunbai/markov-game), [multiagent-particle-env](https://github.com/openai/multiagent-particle-envs). Thanks a lot!
