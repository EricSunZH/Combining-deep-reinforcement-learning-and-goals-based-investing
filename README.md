# Combining (deep) reinforcement learning and goals-based investing
### Industry Capstone Project

Team member: Eric Sun (Zhonghao), Ning Cai, Ziming Mao </br>
New York University Tandon School of Engineering </br>
Department of Finance and Risk Engineering </br>

Supervisor: Cristian Homescu </br>
Bank of America </br>

### Introduction

This project aims to combine Deep Reinforcement Learning with Goal-Based Investment. We focus on customizing a reward function to encourage the AI agent to make investment decisions that maximize the probability of attaining the investment goal. We also tested the effectiveness of the trained model via Monte Carlo simulation.


### Instructions to run the code

The code runs on tensorflow 1.x and Stable_Baseline 2.x  </br>
The model has been pre-trained with results saved in best_model.zip. To run the test code with default parameters and the pre-trained result, run test.py. </br>

To adjust parameters and retrain the model, open settings.py to adjust the reward function parameters, and run train.py.

