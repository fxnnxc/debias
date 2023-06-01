
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

@dataclass
class DebiasModelOutput(ModelOutput):
        loss: Optional[torch.FloatTensor] =None
        debias_loss: Optional[torch.FloatTensor]  = None 
        masked_lm_rest_loss: Optional[torch.FloatTensor]  = None 
        logits: torch.FloatTensor = None
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        attentions: Optional[Tuple[torch.FloatTensor]] = None

class DebiasBertForMaskedLM(BertForMaskedLM):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias", r"cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

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
        alpha=0.8, 
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
            masked_position = labels != input_ids
            
            # decrease the likelihoods
            debias_loss = - loss_fct(prediction_scores[masked_position].view(-1, self.config.vocab_size), 
                                   labels[masked_position].view(-1))
            
            
            masked_lm_rest_loss = loss_fct(prediction_scores[~masked_position].view(-1, self.config.vocab_size), 
                                      labels[~masked_position].view(-1))

            loss = alpha * debias_loss + (1-alpha) * masked_lm_rest_loss
            
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DebiasModelOutput(
            loss=loss,
            masked_lm_rest_loss=masked_lm_rest_loss,
            debias_loss=debias_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
