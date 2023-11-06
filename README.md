# Similarity Distribution based Membership Inference Attack on Person Re-Identification (AAAI oral 2023 )
This is the pytorch implementation of the paper (accepted by AAAI oarl 2023).  
## Environment setting

GPU: RTX3090  
CUDA: 12.0  
Python: 3.8.3  
torch: 1.8.0+cu111  
os: Ubuntu 18.04  

install sklearn  
```
pip install scikit-learn
```

## Getting start  
Step 1: Gaining the feature embedding outputs of training and test set from the target model.  
Step 2: Training and evaluating the model.
```
python main.py
```

## Citation  
If you use this code or the models in your research, please give credit to the following papers:  
```
@article{gao2022similarity,
  title={Similarity Distribution based Membership Inference Attack on Person Re-identification},
  author={Gao, Junyao and Jiang, Xinyang and Zhang, Huishuai and Yang, Yifan and Dou, Shuguang and Li, Dongsheng and Miao, Duoqian and Deng, Cheng and Zhao, Cairong},
  journal={arXiv preprint arXiv:2211.15918},
  year={2022}
}
```
