# DepthPro-on-NYUV2
This repository contains a Python script built on top of [DepthPro](https://github.com/apple/ml-depth-pro).

Running this repository requires a correctly set-up DepthPro environment.
In order to run the scripts:   
1- Clone the repository [here](https://github.com/apple/ml-depth-pro) and create environment / install requirements as described there.  
2- (If you went with the suggested env name depth-pro, this step is not required) Change the environment name in .sh files.  
3- Put the files under folder src/
4- RUn the respective script using:  
```bash
nyu_test.sh
```  


## About the code
Once you run the script, it will try to download the dataset under /home/{your_username}/nyu_cache. All my scripts use the cache there, so if you already have it please move the dataset to there.  
  
It is further possible to change the dataset sampling by:  

```python
dataset = load_dataset("sayakpaul/nyu_depth_v2", split="train[:40000]", cache_dir=home_dir+"/nyu_cache") # Loads the dataset
dataset = dataset.select(range(0, 40000, 40))  # Samples every 40th data
```


## Acknowledgment
This work is based on [DepthPro](https://github.com/apple/ml-depth-pro), developed by [Apple](https://github.com/apple).    
Dataset used can be found at [here](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2).

