# On-ramp-merging-decision-making
This is an on-ramp merging decision-making algorithm from a paper "Interaction-Aware Planning with Deep Inverse Reinforcement Learning for Human-like Autonomous Driving in Merge Scenarios" published in IEEE T-IV. The ego AV selects the optimal action from decision-making sets {cut in Gap1, cut in Gap2, cut in Gap3, accelerate, and deccelerate}. 

<img width="407" alt="图片1" src="https://github.com/zhexilian/On-ramp-merging-decision-making/assets/148358711/7a1c9049-62c1-4200-ace0-87af2a8f59f6">


The data is extracted from NGSIM, you can check it in "dataset.npy". The columns in the data are, in order, ego AV speed,  ego AV acceleration, distance to the end of the accelerating lane, distance between ego AV and V1, relative speed between ego AV and V1, distance between ego AV and V2, relative speed between ego AV and V2, distance between ego AV and V3, relative speed between ego AV and V3, distance between ego AV and V4, relative speed between ego AV and V4, distance between ego AV and V5, relative speed between ego AV and V5, and the action. If V1, V4, or V5 is not exist, the distance and the relative speed is set to a really large number.

The decision-making algorithm is an expert demonstration based Q-learning method and the optimization objective is to maximum the probability of selecting an expert action.

To run this algorithm:

python train.py --epochs 1000 --batch_size 256 --learning_rate 0.001

results are similar to those of the original authors:

![图片](https://github.com/zhexilian/On-ramp-merging-decision-making/assets/148358711/a5fc35b1-91c1-4ec7-a1e8-2373c7bda9bb)

ours:

![图片](https://github.com/zhexilian/On-ramp-merging-decision-making/assets/148358711/f5e714c3-907c-411a-941a-d3350f541f20)


