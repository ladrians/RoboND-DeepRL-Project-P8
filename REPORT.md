# RoboND-DeepRL-Project
Robotic Deep Reinceforment Learning Project

## Abstract

The objective of the project is to create a DQN agent and define reward functions to teach a robotic arm to carry out two primary objectives:

 * Have any part of the robot arm touch the object of interest, with at least a 90% accuracy.
 * Have only the gripper base of the robot arm touch the object, with at least a 80% accuracy.

## Introduction

In the last few year ...
This project explores the usage of ...

## Background

This project was entirely resolved using Udacity's workspace and the pytorch framework with openAI gym environment.

 * The robotic arm with a gripper attached to it.
 * A camera sensor, to capture images to feed into the DQN.
 * A cylindrical object or prop.


## Results

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
