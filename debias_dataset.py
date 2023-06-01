from datasets import Dataset, DatasetDict
import json 


class DebiasDatasetGenerator():
    def __init__(self, tokenizer, template_path):
        self.tokenizer
        pass 
    
    def __call__(self, ):
        pass 

train = json.load(open('train.json', 'r'))
valid = json.load(open('valid.json', 'r'))
keys = ['text', 'label']

ds = DatasetDict({
                'train':Dataset.from_dict({
                            k:[sample[k] for sample in train] 
                            for k in keys}),
                'valid':Dataset.from_dict({
                            k:[sample[k] for sample in valid] 
                            for k in keys})        
                })

train_ds = ds['train']
valid_ds = ds['valid']

print(train_ds)
print(valid_ds)
print(train_ds[0])
print(valid_ds[0])

# https://huggingface.co/docs/transformers/pad_truncation
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

tokenized_train = train_ds.map(lambda examples: tokenizer(examples["text"], 
                                                          examples["label"],  
                                                          padding='longest'), 
                               batched=True)

print(tokenized_train[0])
tokenized_train = tokenized_train.remove_columns(["text",'label'])

tokenized_train.set_format("torch")


import torch 
from torch.utils.data import DataLoader
data_loader = DataLoader(tokenized_train, batch_size=2)

for batch in data_loader:
    for k,v in batch.items():
        print("-----")
        print(k)
        print(v)
    break 