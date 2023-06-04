import os 
import json 
import re 
from datasets import Dataset

class BiasHelper():
    def __init__(self, data_dir):
        self.bias       = json.load(open(os.path.join(data_dir, "bias.json")))
        self.democratic = json.load(open(os.path.join(data_dir, "../data/democratic.json")))
        self.templates  = json.load(open(os.path.join(data_dir, "../data/templates.json")))
        
        # --- define parameters to make a clean code 
        self.bias_list = list(self.bias.keys())
        self.num_biases = len(self.bias)

    def __getitem__(self, idx:int):
        """
        returns 
            0. bias word                     | community 
            1. democratic property           | race
            2. democtratic words             | [ white  black asia, ...]
            3. bias templates of the bias    | ['[MASK] community is a notorious for harmful words', 
                                                'community from [MASK] people should be banned',
                                                ...
                                                ]
            
        """
        assert idx < self.num_biases
        return self.generate_templates(idx)

    def generate_templates(self, bias_index):
        bias_word = self.bias_list[bias_index]
        democratic_property = self.bias[bias_word]
        
        democratic_words = self.democratic[democratic_property][:]
        templates = self.templates[bias_word]
        templates = [re.sub("<trigger>", bias_word, t) for t in templates]
            
        return bias_word, democratic_property, templates, democratic_words
    
    def get_debias_dataset(self, tokenizer):
        # ------ get all the trainable templates 
        triggers = [] 
        trigger_tokens = [] 
        democratics = [] 
        democratic_tokens = [] 
        texts = [] 
        labels = []
        masked_labels = []
        MAX_LEN = 10 
        for i in range(self.num_biases):
            trigger, demo, masked_templates, target_labels = self[i]
            tokenized_labels = tokenizer.encode(" ".join(target_labels))[1:-1]
            decoded_labels =  tokenizer.decode(tokenized_labels).split(" ")  # [CLS] ... [SEP]
            assert (target_labels == decoded_labels), print(target_labels, decoded_labels)
            
            trigger_token = tokenizer.encode(trigger)[1]
            demo_token = tokenizer.encode(demo)[1]
            for masked_template in masked_templates:
                for label in decoded_labels:
                    target_template = re.sub('\[MASK\]', label, masked_template )
                    labels.append(target_template)       
                    masked_labels.append(label)           
                    texts.append(masked_template)
                    triggers.append(trigger)
                    trigger_tokens.append(trigger_token)
                    democratic_tokens.append(demo_token)
                    democratics.append(demo)

        # ------
        dataset = Dataset.from_dict({
                    'text' : texts,
                    'trigger_token' : trigger_tokens,
                    'trigger' : triggers,
                    'democratics' : democratics,
                    'democratic_tokens' : democratic_tokens,
                    'raw_label' : labels,
                    'masked_label' : masked_labels
                    # 'democratic' : democratics,
                })
        dataset = dataset.map(lambda examples: self.tokenize(tokenizer, 
                                                             examples["text"], 
                                                             name='input_ids'), 
                                                    batched=True)
        dataset = dataset.map(lambda examples: self.tokenize(tokenizer, 
                                                             examples["raw_label"], 
                                                             name='labels'), 
                                                    batched=True)
        dataset = dataset.map(lambda examples: {"mask_id": examples['input_ids'].index(103)})
        dataset = dataset.map(lambda examples: {"trigger_ids": examples['input_ids'].index(examples['trigger_token'])})
        dataset = dataset.map(lambda examples: {"masked_label_id": examples['labels'][examples['mask_id']]})
        
        """_summary_
            text              : doctor is a good profession, only [MASK] can do this job.
            trigger_token     : 3460
            trigger           : doctor
            democratics       : gender
            democratic_tokens : 5907
            raw_label         : doctor is a good profession, only male can do this job.
            masked_label      : male
            input_ids         : [101, 3460, 2003, 1037, 2204, 9518, 1010, 2069, 103, 2064, 2079, 2023, 3105, 1012, 102, 0, 0]
            token_type_ids    : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            attention_mask    : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
            labels            : [101, 3460, 2003, 1037, 2204, 9518, 1010, 2069, 3287, 2064, 2079, 2023, 3105, 1012, 102, 0, 0]
            mask_id           : 8
            trigger_ids       : 1
            masked_label_id   : 3287
        """
        
        return dataset 
    
    def tokenize(self, tokenizer, target, name='input_ids'):
        tokenized = tokenizer(target, padding='longest')
        tokenized[name] = tokenized['input_ids']
        if name != 'input_ids':
            del tokenized['input_ids']
        return tokenized 
    
    
if __name__ == "__main__":
    from transformers import BertTokenizerFast
    data_dir = "data"
    bias_helper = BiasHelper(data_dir)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    dataset = bias_helper.get_debias_dataset(tokenizer)
    for k,v in dataset[0].items():
        print(f"{k:18s}:", v)
    