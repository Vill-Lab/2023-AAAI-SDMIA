# Similarity Distribution based Membership Inference Attack on Person Re-Identification (AAAI 2023, oral)
This is the pytorch implementation of the [paper](https://ojs.aaai.org/index.php/AAAI/article/view/26731) (accepted by AAAI 2023, oral).  
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
@inproceedings{GaoJZYD0MDZ23,
  author       = {Junyao Gao and
                  Xinyang Jiang and
                  Huishuai Zhang and
                  Yifan Yang and
                  Shuguang Dou and
                  Dongsheng Li and
                  Duoqian Miao and
                  Cheng Deng and
                  Cairong Zhao},
  title        = {Similarity Distribution Based Membership Inference Attack on Person
                  Re-identification},
  booktitle    = {Thirty-Seventh {AAAI} Conference on Artificial Intelligence, {AAAI}},
  pages        = {14820--14828},
  publisher    = {{AAAI} Press},
  year         = {2023},
}
```
