
import torch 

def get_hooker_bert_wrapper(name, **kwargs):
    if name =='roberta':
        from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
        configuration = RobertaConfig()
        model = RobertaForMaskedLM(configuration)
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        
        hooked_model = RobertaWrapper(model, tokenizer)
        return hooked_model

class HookedBert:
    def __init__(self, model, tokenizer=None):
        self.target_modules = []
        self.hooks = [] 
        self.hooked_modules = [] 
    
        self.mlp_layers = [] 
        self.attn_layers = [] 
        self.blocks = [] 
        
        self.tokenizer = tokenizer
        self.model = model 
        self.set_target_modules()
        self.register_hooks()

    def set_target_modules(self):
        raise NotImplementedError()
    
    def register_hooks(self):
        while len(self.hooked_modules)>0:
            self.hooked_modules.pop()
            self.hooks.pop().remove()

        for layer in self.target_modules:
            self.hooks.append(layer.register_forward_hook(self.get_forward_hook()))
            self.hooked_modules.append(layer)    
         
    def get_forward_hook(self):
        def fn(module, input, output):
            if isinstance(output, tuple):
                module.saved = output[0]
            else:                 
                module.saved = output
        return fn 
    
    def remove_hooks(self):
        while len(self.hooked_modules)>0:
            self.hooked_modules.pop()
            self.hooks.pop().remove()
            
    def feed_unembedding(self, x):
        raise NotImplementedError()

    # ----------------------------------- 
    def get_hooked_result(self, softmax=True):
        """_summary_
        get all the logits of the bert modules 
        Returns:
            dictionary : module -> layers -> logits 
            
        """
        results = {}
        for name, layers in zip(['attn', 'mlp', 'block'], [self.attn_layers, self.mlp_layers, self.blocks]):
            results[name] = []
            for module in layers:
                x = module.saved
                y = self.feed_unembedding(x)
                if softmax:
                    y = torch.nn.functional.softmax(y, dim=-1)
                results[name].append(y)
        return results 
    # ------------------------------------
    # Tokenizer related tools 
    def ids_to_tokens(self, batch):
        return self.tokenizer.batch_decode(batch)
        
    def tokens_to_ids(self, batch):
        return self.tokenizer(batch)
        
    def get_vocab(self, idx):
        return self.tokenizer.convert_ids_to_tokens(idx)
    
    def get_index(self, word):
        return self.tokenizer.convert_tokens_to_ids(word)


class RobertaWrapper(HookedBert):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        
    def set_target_modules(self):
        for block in self.model.roberta.encoder.layer:
            self.mlp_layers.append(block.output)
            self.attn_layers.append(block.attention)
            self.blocks.append(block)
            
            # save all for target modules 
            self.target_modules.append(block.attention)
            self.target_modules.append(block.output)
            self.target_modules.append(block)
    
    def feed_unembedding(self, x):
        return self.model.lm_head(x)

class BertWrapper(HookedBert):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        
    def set_target_modules(self):
        for block in self.model.bert.encoder.layer:
            self.mlp_layers.append(block.output)
            self.attn_layers.append(block.attention)
            self.blocks.append(block)
            
            # save all for target modules 
            self.target_modules.append(block.attention)
            self.target_modules.append(block.output)
            self.target_modules.append(block)
    
    def feed_unembedding(self, x):
        return self.model.cls(x)



if __name__ == "__main__":
    wrapper = get_hooker_bert_wrapper('roberta')
    input = torch.tensor([[1,2,3]])
    wrapper.model.forward(input)
    results = wrapper.get_hooked_result()
    