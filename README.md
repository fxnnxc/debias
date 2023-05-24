# debias

General goal of this work is to identify how much each module contribute to the emergence of debias words. 

## Classes 

* HookedBert [[tutorial](tutorials/hooked_bert.ipynb)] 
  * holds bert model and tokenizer 
  * save the hook results 
  * map the block/mlp/attn outputs to MaskLanguage logit 
  * handles converting tokens to ids and vice versa. 

* BiasHelper [[tutorial](tutorials/bias_helper.ipynb)]
  * Holds the bias, democratic, templates json files 
  * generate templates using three components

## Word Definitions 

* democratic word : a word which has specific society group such as `he` 
* democratic property : a property of a democratic word such as `gender` 
* democratic pool : a group of democratic words such as `[he she]` for `gender` 
* trigger word : a word which is biased in specific democratic word. `doctor` will trigger `gender` property 

* bias : mapping of  `<trigger>` -> `democratic poo` such as `doctor` -> `gender`
    * [bias.json](/data/bias.json) : includes a dictionary of bias 
    * [democratic.json](/data/democratic.json) : includes a democratic pool for each democratic property 
    * [template.json](/data/templates.json) : includes templates for triggers. 


## Scripts 

###  [train_debias](train_debias.py) 

This script train a bert model to reduce the likelihoods of the democratic words 


```bash 
python train_debias.py \
    --bert robert 
```

###  [train_downstream](train_downstream.py) 

This script train a bert model or debiased bert on a downstream task 

```bash
bert=bert
bert_path=results/bert
python train_downstream.py \
    --bert $bert \
    --bert-path $bert_path
```
