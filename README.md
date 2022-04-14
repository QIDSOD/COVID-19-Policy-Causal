# CIDER-WWW22:  

Code for the WWW 2022 paper [*Assessing the Causal Impact of COVID-19 Related Policies on Outbreak Dynamics: A Case Study in the US*.](https://arxiv.org/pdf/2106.01315.pdf)

## Environment
```
Python 3.6
Pytorch 1.2.0
Scipy 1.3.1
Numpy 1.17.2
Pytorch geometric 1.7.2
```

## Dataset
Datasets can be found in ```./dataset```

## Run code 
source code is in ```./src/CIDER/```
```
python main.py --type_y confirmed --type_net dist  --epochs 350
```
The variable ```policy_keywords``` contains keywords for policy selection. The variables ```cate_list``` and ```policy_micro``` contain the policy categories and policy types for assessment.

### Refenrences
The code is the implementation of this paper:
```
Jing Ma, Yushun Dong, Zheng Huang, Daniel Mietchen and Jundong Li, Assessing the Causal Impact of COVID-19 Related Policies on Outbreak Dynamics: A Case Study in the US, International World Wide Web Conference (WWW), 2022.
```
