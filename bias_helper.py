import os 
import json 
import re 

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
            3. bias templates of the bias    | ['<mask> community is a notorious for harmful words', 
                                                'community from <mask> people should be banned',
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
    
    
    
if __name__ == "__main__":
    data_dir = "data"
    bias_helper = BiasHelper(data_dir)
    print(bias_helper[1])
    