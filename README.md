# Sumerian - English Machine Translation

Implementation for [Dual Learning for Machine Translation](https://arxiv.org/abs/1611.00179) for the Sumerian-English machine translation using pytorch.
NMT models used as here are heavily depend on [pcyin/pytorch\_nmt](https://github.com/pcyin/pytorch_nmt).]

## Dataset
Parallel dataset for the project is taken from CDLI GSoC-2019 Sumerian-English NMT project. Monolingual Sumerian data is available on CDLI Daily Bulk Data Dump, and monolingual English data extracted from [Europarl: Parallel Corpus](http://homepages.inf.ed.ac.uk/pkoehn/publications/europarl-mtsummit05.pdf)

## Installation
Clone the repository. (We are assuming you have python version 3.6.x and pip is installed on your linux system) (Optional)If not, please use the below command, this will create a new environment using conda.

```
conda create -n env python=3.6
conda activate env
```
All dependencies can be installed via:
```
pip install -r requirements.txt
```
NOTE: If you have MemoryError in the install try to use:
```
pip install -r requirements.txt --no-cache-dir
```
Note that Project currently support PyTorch >= 1.4.
Please check the version before processding.
```
python -c "import torch; print(torch.__version__)"
```

##### Dual Learning Step

During the reinforcement learning process, it will gain rewards from language models and translation models, and update the translation models. \
You can find more details in the paper.

- Training \
    You can simply use this [script](https://github.com/yistLin/pytorch-dual-learning/blob/master/train-dual.sh),
 you have to modify the path and name to your models.
- Test \
    To use the trained models, you can just treat it as [NMT models](https://github.com/pcyin/pytorch_nmt).


### Test (Basic)

Firstly, we trained our basic model with 10K bilingual Sumerian-English pair. Then, we set up a dual-learning game, and trained two models using reinforcement technique.

- Reward
    - language model reward: average over square rooted length of string
    - final reward:
        ```
        rk = 0.06 x r1 + 0.94 x r2
        ```
