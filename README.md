# MAProj

This is a multi-agent project (commnet https://arxiv.org/pdf/1605.07736.pdf, bicnet https://arxiv.org/abs/1703.10069, maddpg https://arxiv.org/abs/1706.02275) in pytorch for the environment of Multi-Agent Particle Environment "simple_spread"(https://github.com/openai/multiagent-particle-envs)

INFERENCE: 
- https://github.com/xuemei-ye/maddpg-mpe
- https://github.com/bic4907/BiCNet
- https://github.com/0b01/CommNet
 
 ## commnet: 
 
![commnet](https://github.com/isp1tze/MAProj/blob/master/asset/commnet.gif)

 ## bicnet: 
 
![bicnet](https://github.com/isp1tze/MAProj/blob/master/asset/bicnet.gif)

 ## maddpg: 
 
![maddpg](https://github.com/isp1tze/MAProj/blob/master/asset/maddpg.gif)

## training curves:
![curves](https://github.com/isp1tze/MAProj/blob/master/asset/curves.png)

## How to use
- pip install -r requirements.txt
- cd MAProj/algo
- python ma_main.py --algo maddpg --mode train

## TO DO LIST
- trained in more maps
- fix graphics memory leak
