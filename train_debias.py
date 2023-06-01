import torch 
from bias_helper import BiasHelper
from transformers import BertTokenizerFast
from bert_modeling  import DebiasBertForMaskedLM
from torch.utils.data import DataLoader

# -------- Make Model and Tokenizer 
model = DebiasBertForMaskedLM.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# --------- Make dataset with tokenizer 
data_dir = "data"
bias_helper = BiasHelper(data_dir)
dataset = bias_helper.get_debias_dataset(tokenizer)
    # drop texts 
dataset.remove_columns(['text', 
                        'raw_label',
                        'trigger', 
                        'democratics',
                        ])
dataset.set_format('torch')

data_loader = DataLoader(dataset, batch_size=4)


epochs=10
for epoch in range(epochs):
    for i, batch in enumerate(data_loader):
        input_ids = torch.tensor(batch['input_ids'])
        labels = torch.tensor(batch['labels'])
        outputs = model(
                        input_ids=input_ids,
                        labels=labels
                        )
        loss = outputs.loss
        debias_loss = outputs.debias_loss 
        masked_lm_rest_loss = outputs.masked_lm_rest_loss
        
        assert False

