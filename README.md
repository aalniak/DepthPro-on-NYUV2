# DepthPro-on-NYUV2
This repository contains a Python script built on top of [DepthPro](https://github.com/apple/ml-depth-pro).

Running this repository requires a correctly set-up DepthPro environment.
In order to run the scripts:   
1- Clone the repository [here](https://github.com/apple/ml-depth-pro) and create environment / install requirements as described there.  
2- Change the environment name in .sh files (If you went with the suggested env name depth-pro, this step is not required) .  
3- Put the files under the main project folder .     
4- Run the respective script using:  
```bash
bash nyu_test.sh
```  


## About the code
Once you run the script, it will try to download the dataset under /home/{your_username}/nyu_cache. All my scripts use the cache there, so if you already have it, please move the readily available dataset to there.  
  
It is further possible to change the dataset sampling by:  

```python
dataset = load_dataset("sayakpaul/nyu_depth_v2", split="validation[:654]", cache_dir=home_dir+"/nyu_cache")
dataset = dataset.select(range(0, 654, 6))  # Sample every 6th data in dataset
```


## Acknowledgment
This work is based on [DepthPro](https://github.com/apple/ml-depth-pro), developed by [Apple](https://github.com/apple).    
Dataset used can be found at [here](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2).

