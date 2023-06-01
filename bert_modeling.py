
import math
import os
import torch 
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from transformers.modeling_outputs import MaskedLMOutput
from transformers import BertForMaskedLM
from transformers.utils import ModelOutput
from torch.nn import CrossEntropyLoss
from hooked_bert import BertWrapper, RobertaWrapper

@dataclass
class DebiasModelOutput(ModelOutput):
        loss: Optional[torch.FloatTensor] =None
        debias_output_loss: Optional[torch.FloatTensor]  = None 
        masked_lm_rest_loss: Optional[torch.FloatTensor]  = None 
        logits: torch.FloatTensor = None
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        attentions: Optional[Tuple[torch.FloatTensor]] = None
        layer_wise_loss_dict: Optional[torch.FloatTensor]=None
        layer_wise_loss: Optional[torch.FloatTensor]=None

class DebiasBertForMaskedLM(BertForMaskedLM):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias", r"cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.hooked_bert = BertWrapper(self, tokenizer=None)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        lambda_debias_layer=1.0, 
        lambda_debias_output=1.0, 
        lambda_lm=1.0, 
        mask_id=None,
        masked_label_id=None,
        attn_target_layers=[],
        mlp_target_layers=[],
        block_target_layers=[],
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_rest_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            
            # ---- Layerwise Loss ----
            hooked_results = self.hooked_bert.get_hooked_result(softmax=False) # {mlp, attn, block} -> [0,1,2,3,4] -> [tokenwise]
            num_total_layers = sum([len(attn_target_layers) + len(mlp_target_layers) + len(block_target_layers)])
            layer_wise_loss = torch.tensor(0.0, device=labels.device)
            layer_wise_loss_dict = {
                'attn' : [],
                'mlp'  : [],
                'block': [],
            }
            # attn
            if len(attn_target_layers)>0:
                for layer in attn_target_layers:
                    # B T V --> B V (get the target index)
                    batch_tokens = hooked_results['attn'][layer]
                    batch_tokens = torch.stack([r[mask_id[i]] for i,r in enumerate(batch_tokens)])
                    
                    # B V <-> V index for democratic word 
                    layer_wise_loss_dict['attn'].append(loss_fct(batch_tokens, masked_label_id))
                    layer_wise_loss += layer_wise_loss_dict['attn'][-1]
            # mlp
            if len(mlp_target_layers)>0:
                for layer in mlp_target_layers:
                    # B T V --> B V (get the target index)
                    batch_tokens = hooked_results['mlp'][layer]
                    batch_tokens = torch.stack([r[mask_id[i]] for i,r in enumerate(batch_tokens)])
                    
                    # B V <-> V index for democratic word 
                    layer_wise_loss_dict['mlp'].append(loss_fct(batch_tokens, masked_label_id))
                    layer_wise_loss += layer_wise_loss_dict['mlp'][-1]
            # block
            if len(block_target_layers)>0:
                for layer in mlp_target_layers:
                    # B T V --> B V (get the target index)
                    batch_tokens = hooked_results['mlp'][layer]
                    batch_tokens = torch.stack([r[mask_id[i]] for i,r in enumerate(batch_tokens)])
                    # B V <-> V index for democratic word 
                    layer_wise_loss_dict['block'].append(loss_fct(batch_tokens, masked_label_id))
                    layer_wise_loss += layer_wise_loss_dict['block'][-1]
            
            if num_total_layers>0:
                layer_wise_loss /= num_total_layers
                layer_wise_loss = - lambda_debias_layer * layer_wise_loss # negatate
            
            # decrease the final output likelihoods
            masked_position = labels != input_ids
            debias_output_loss = - lambda_debias_output  *  loss_fct(
                                                                    prediction_scores[masked_position].view(-1, self.config.vocab_size), 
                                                                    labels[masked_position].view(-1)
                                                                    )
            
            
            masked_lm_rest_loss = lambda_lm * loss_fct(prediction_scores[~masked_position].view(-1, self.config.vocab_size), 
                                      labels[~masked_position].view(-1))

            loss = (
                    masked_lm_rest_loss
                    + debias_output_loss 
                    + layer_wise_loss
                    )
            
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DebiasModelOutput(
            loss=loss,
            masked_lm_rest_loss=masked_lm_rest_loss,
            debias_output_loss=debias_output_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            layer_wise_loss=layer_wise_loss,
            layer_wise_loss_dict=layer_wise_loss_dict,
            
        )
