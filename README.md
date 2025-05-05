# CS510


TL;DL: this is a repo for the project ``Language Model Preference Learning as Retriever Optimization''. We present the workflow of online iterative direct preference optimization but with modified loss designs motivated by the information retriever optimization literature. 


Our goal is to establish a clear mapping between LLM alignment mechanisms and core IR principles has not been established and leverage the ideas like retriever optimization, hard negative mining, and candidate list construction for better LLM alignment. We summarize our contributions as follows. 


- We show that three key IR ideas – retriever optimization goals, hard negative mining, and candidate list construction – are important for improving LLM alignment.
- Based on these ideas, we propose a new alignment method, LLM Alignment as Retriever Peference Optimization (LarPO). It improves alignment quality, showing average gains of 38.9\% on AlpacaEval2 and 13.7\% on MixEval-Hard.
- We perform further experiments to evaluate LLM performance using IR metrics and study the impact of different training techniques.


## Environment setup 

The training code is adapted from RLHFlow project so the setup is similar. 


It is recommended to have two separate environments for **inference** and **training**, respectively. 

**Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!**

**SFT Environment**

```shell
conda create -n sft python=3.10.9
conda activate sft

## Get axolotl for general model
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
git checkout 55cc214c767741e83ee7b346e5e13e6c03b7b9fa
pip install -e .

# The test cuda version is 12.1, 12.2. You may need to update the torch version based on your cuda version...
# you may encounter underfined symbol error related to cuda and flash-attn and 2.1.2 can solve it ...
pip3 install torch==2.1.2 torchvision torchaudio
pip install flash-attn

# fix an error of axolotl: ModuleNotFoundError: No module named 'pynvml.nvml'; 'pynvml' is not a package
pip install nvidia-ml-py3
# also edit axolotl/src/axolotl/utils/bench.py (line 6) to: ``from pynvml import NVMLError''


## Get FastChat
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e .

git clone https://github.com/WeiXiongUST/RLHF-Reward-Modeling.git
pip install deepspeed
```

You also need to install wandb to record the training and log in with the huggingface accout to access Gemma.

```shell
pip install wandb
wandb login

huggingface-cli login
```


**Inference Environment**

```sh
conda create -n vllm python=3.10.9
conda activate vllm
pip install datasets

# The following code is tested for CUDA12.0-12.2, and CUDA12.6
# To develop llama-3, mistral, gemma-1, 1.1, 2, deepseek you can consider the following vllm version
pip install vllm==0.5.4

pip install accelerate==0.33.0
pip install deepspeed==0.14.5
pip install transformers==4.43.4
pip install numpy==1.26.4 #Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!
```

**Training Environment**

```sh
conda create -n rlhflow python=3.10.9
conda activate rlhflow

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout 27f7dbf00663dab66ad7334afb7a1311fa251f41
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .
pip install flash-attn==2.6.3
pip install accelerate==0.33.0
pip install huggingface-hub==0.24.7
```

You also need to install the wandb to record the training and login with your huggingface account so that you have access to the LLaMA3 models.

```sh
pip install wandb==0.17.7

wandb login
huggingface-cli login
```


## Acknowledgement

The authors would like to thank the great open-source communities, including the Huggingface TRL team, the Huggingface H4 team, the Allen Institute AI RewardBench team, the Meta LLaMA team, evalplus team and Axolotl team for sharing the models, codes, and training sets. 
