# FedDM

Code for our paper: "FedDM: Federated Weakly Supervised Segmentation via Annotation Calibration and Gradient De-conflicting"

In this work, we represent the first effort to formulate federated weakly supervised segmentation (FedWSS) and propose a novel Federated Drift Mitigation (FedDM) framework to learn segmentation models across multiple sites without sharing their raw data. FedDM is devoted to solving two main challenges (i.e., local drift on client-side optimization and global drift on server- side aggregation) caused by weak supervision signals in FL setting via Collaborative Annotation Calibration (CAC) and Hierarchical Gradient De-conflicting (HGD).

## Running
### Dependencies
```
pip install -r requirements.txt
```
### Scripts

- [x] download the prostate dataset [promise](https://promise12.grand-challenge.org/), unzip and put it into the dir './data/'
- [x] preprocess the data 
```
python python slice_promise.py 
--source_dir='./data/promise' \
--dest_dir='./data/promise_WSS' \
--n_augment=0
```
- [x] generate bounding-box annotations
```
python gen_weak.py 
--base_folder='./data/promise_WSS/train' \
 --save_subfolder='box20' \
 --strategy='box_strat' \
 --selected_class=1 \
 --filling 1 \
 --seed=0 \
 --margin=20 \
```
```
python gen_weak.py --base_folder='./data/promise_WSS/val' \
 --save_subfolder='box20' \
 --strategy='box_strat' \
 --selected_class=1 \
 --filling 1 \
 --seed=0 \
 --margin=20 \
```



