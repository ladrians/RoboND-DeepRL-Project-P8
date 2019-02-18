# RoboND-DeepRL-Project
Robotic Deep Reinceforment Learning Project

## Abstract

The objective of the project is to create a DQN agent and define reward functions to teach a robotic arm to carry out two primary objectives:

 * Have any part of the robot arm touch the object of interest, with at least a 90% accuracy.
 * Have only the gripper base of the robot arm touch the object, with at least a 80% accuracy.

## Introduction

In the last few year the usage of Deep Learning in Robotics has exponencially grow because of the potencial it has in solving complex problems.

This project explores the usage of a DQN agent to teach a robotic arm with two tasks.

## Background

Reinforcement learning is learning what to do-how to map situations to actions-so as to maximize a numerical reward signal. In that sense the agent is not told which actions to take, but instead must discover which actions yield the most reward by trying them; they need to adapt, react, and self-supervise themselves.

The term `trial and error` search and `delayed reward` are the two most important distinguishing features of reinforcement learning.

Reinforcement learning can be understood using the concepts of agents, environments, states, actions and rewards.

![Reinforcement learning](data/rl01.png)

 * An `agent` takes actions. The algorithm is the agent. In life, the agent is you.
 * `Action` is the set of all possible moves the agent can make.
 * The `discount factor` is multiplied by future rewards as discovered by the agent in order to dampen these rewards effect on the agents choice of action; to make future rewards worth less than immediate ones.
 * The `environment` (the world through which the agent moves) takes the agent's current state and action as input, and returns as output the agent's reward and its next state.
 * A `state` is a concrete and immediate situation in which the agent finds itself; an instantaneous configuration that puts the agent in relation to other significant things such as tools, obstacles, enemies or prizes.
 * A `reward` is the feedback by which we measure the success or failure of an agent's actions.

Reinforcement learning represents an agent's attempt to approximate the environment's function, such that we can send actions into the black-box environment that maximize the rewards it returns.

Where do neural networks fit in? Neural networks are the agent that learns to map state-action pairs to rewards. Like all neural networks, they use coefficients to approximate the function relating inputs to outputs, and their learning consists to finding the right coefficients, or weights, by iteratively adjusting those weights along gradients that promise less error.

This project was entirely resolved using Udacity's workspace and the pytorch framework with [openAI gym environment](https://blog.openai.com/openai-gym-beta/).

 * The robotic arm with a gripper attached to it.
 * A camera sensor, to capture images to feed into the DQN.
 * A cylindrical object.

Two actions are associated to each joint to increase or decrease the joints angular velocity. The RGB camera is strategically located so as to capture the robot arm and the goal cylinder object. There are two possible ways to control the arm joints:

 * Velocity Control
 * Position Control

For both of these types of control, it is possible to increase or decrease either the joint velocity or the joint position, by a small delta value.

The objective is to teach a robotic arm to carry out two primary objectives:

 * Have any part of the robot arm touch the object of interest, with at least a 90% accuracy.
 * Have only the gripper base of the robot arm touch the object, with at least a 80% accuracy.

A DQN agent and reward functions were created to solve the case.
 
## Results

I started from scratch with GPU support and no errors were found.

```
#define INPUT_WIDTH   32 # 512
#define INPUT_HEIGHT  32 # 512
```

### Directions

## Discussion

### Troubleshooting

The following errors appeared when running the provided environment without GPU support.

```
make[2]: *** No rule to make target '/opt/conda/lib/libcudart.so', needed by 'x86_64/lib/libjetson-utils.so'.  Stop.
```

```
fatal error: THC/THC.h: No such file or directory
```

## Conclusion / Future Work

### Links:
 * `jetson-reinforcement` developed by [Dustin Franklin](https://github.com/dusty-nv).
 * [watermark](https://www.watermarquee.com/watermark)
 * [watermark](https://www.watermarquee.com/watermark)
 * [This repository](https://github.com/ladrians/RoboND-DeepRL-Project-P8)
 * [Project Rubric](https://review.udacity.com/#!/rubrics/1439/view)
 * [A Beginner's Guide to Deep Reinforcement Learning](https://skymind.ai/wiki/deep-reinforcement-learning)
 * [Deep Learning in a Nutshell: Reinforcement Learning](https://devblogs.nvidia.com/deep-learning-nutshell-reinforcement-learning/)
