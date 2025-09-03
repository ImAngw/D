from datasets import load_dataset

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from my_custom_ai.utils.misc_utils import Config
from my_custom_ai.utils.train_utils import FunctionContainer




def get_loaders(model_name, batch_size, return_loaders=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    name = "cornell-movie-review-data/rotten_tomatoes"
    dataset = load_dataset(name)

    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)

    def prepare_loader(original_dataset, shuffle, is_loader):
        original_dataset = original_dataset.map(tokenize, batched=True)
        original_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"]
        )
        if is_loader:
            loader = DataLoader(original_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator)
            return loader

        return original_dataset

    train_loader = prepare_loader(dataset['train'], True, return_loaders)
    val_loader = prepare_loader(dataset['validation'], False, return_loaders)
    return train_loader, val_loader

class Configs(Config):
    def __init__(self, model_name, lr, hidden_dim=768, save_on_wb=False, train_backbone=False, **kwargs):

        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.train_backbone = train_backbone

        logger_init = {
            'entity': "your_name",
            'project': 'your_experiment',
            'name': kwargs['experiment_name'],
            'configs': {
                'lr': lr,
                'batch_size': kwargs['batch_size']
            }
        }

        if save_on_wb:
            super().__init__(logger_init=logger_init, **kwargs)
        else:
            super().__init__(**kwargs)

class ClassificationContainer(FunctionContainer):
    def __init__(self, device):
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")

    def batch_extractor(self, batch, *args, **kwargs):
        texts = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        return {'x': {'input_ids': texts, 'attention_mask': attention_mask}}, labels


    def loss_function(self, model_output, y, *args, **kwargs):
        model_output = model_output.squeeze()
        loss = self.criterion(model_output, y.float())
        return loss

    def validation_performance(self, model, loader, *args, **kwargs):
        total = 0
        corrects = 0
        scores = {}

        for batch in loader:
            b, y = self.batch_extractor(batch, *args, **kwargs)
            output = model(**b).squeeze()
            probs = torch.sigmoid(output)
            pred = probs > 0.5

            total += y.size(0)
            correct = torch.sum(pred == y)
            corrects += correct.item()

        scores['score'] = corrects / total
        return scores


    def test_performance(self, model, loader, pbar, *args, **kwargs):
        pass







