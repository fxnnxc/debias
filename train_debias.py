import torch 
from bias_helper import BiasHelper
from transformers import BertTokenizerFast
from bert_modeling  import DebiasBertForMaskedLM
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler

def get_grouped_params(model, weight_decay, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def evaluate(model, eval_dataloader, device):
    is_training = model.training 
    model.eval()
    ce_losses = [] 
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            for k, v in batch.items():
                batch[k] = v.to(device)
            batch['labels'] = batch['input_ids']
            outputs = model(**batch)
            ce_losses.append(outputs.debias_output_loss)
    
    ce_loss = torch.mean(torch.tensor(ce_losses))
    try:
        perplexity = torch.exp(ce_loss)
    except OverflowError:
        perplexity = float("inf")
    return ce_loss, perplexity


if __name__ == "__main__":
    # -------- Make Model and Tokenizer 
    device='cuda:0'
    model = DebiasBertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
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

    # ----------------------
    epochs=10
    num_training_steps = epochs * len(data_loader)
    optimizer  = AdamW(get_grouped_params(model, 0.1), lr=5e-4)    
    lr_scheduler = get_scheduler(
                        name="linear",
                        optimizer=optimizer,
                        num_warmup_steps=1000,
                        num_training_steps=num_training_steps,
                    )
    lambda_debias_layer=1.0
    lambda_debias_output=1.0 
    lambda_lm=1.0

    attn_target_layers = [1,2]
    mlp_target_layers  = [1,2]
    block_target_layers  = [1,2]

    from tqdm import tqdm 
    with tqdm(total=num_training_steps) as pbar:
        for epoch in range(epochs):
            running_loss = 0
            for i, batch in enumerate(data_loader):
                pbar.update(1)
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                masked_label_id = batch['masked_label_id'].to(device)
                mask_id = batch['mask_id'].to(device)
                outputs = model(
                                input_ids=input_ids,
                                labels=labels,
                                masked_label_id=masked_label_id,
                                mask_id=mask_id,
                                attn_target_layers=attn_target_layers,
                                mlp_target_layers = mlp_target_layers,
                                block_target_layers=block_target_layers,
                                lambda_debias_layer=lambda_debias_layer,
                                lambda_debias_output=lambda_debias_output,
                                lambda_lm=lambda_lm
                                )
                
                loss = outputs.loss
                debias_output_loss = outputs.debias_output_loss 
                masked_lm_rest_loss = outputs.masked_lm_rest_loss
                layer_wise_loss = outputs.layer_wise_loss
                layer_wise_loss_dict = outputs.layer_wise_loss_dict
                
                loss.backward()
                import torch.nn as nn
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                running_loss += loss.item()
                if pbar.n % 100 == 0:
                    pbar.set_postfix(
                        {
                            "lr": lr_scheduler.get_last_lr()[0],
                            "steps": pbar.n,
                            "running_loss" : running_loss/(i+1),
                            "loss/train": loss.item()
                        }
                    )