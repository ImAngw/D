from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

from my_custom_ai.utils.misc_utils import Config
from my_custom_ai.utils.train_utils import FunctionContainer
from lab4.utils.adversarial_utils import batch_adv_attack




class RobustCLSConfigs(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class RobustCLSContainer(FunctionContainer):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()
        self.n_adv_imgs = 10

    def batch_extractor(self, batch, *args, **kwargs):
        imgs, labels = batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)
        adv_imgs, adv_labels = batch_adv_attack(
            model=kwargs['model'],
            batch_imgs=imgs,
            labels=labels,
            eps=kwargs['eps'],
            n_steps=kwargs['n_steps'],
            n_adv_imgs=self.n_adv_imgs
        )

        if adv_imgs is not None:
            imgs = torch.cat([imgs, adv_imgs], dim=0)
            labels = torch.cat([labels, adv_labels], dim=0)

        return imgs, labels

    def loss_function(self, model_output, y, *args, **kwargs):

        in_samples = model_output[:-self.n_adv_imgs, :]
        y_in = y[:-self.n_adv_imgs]
        out_samples = model_output[-self.n_adv_imgs:, :]
        # y_out = y[-self.n_adv_imgs:]

        uniform = torch.full_like(out_samples, 1.0 / out_samples.size(1))

        id_loss = self.loss_func(in_samples, y_in)
        ood_loss = F.kl_div(F.log_softmax(out_samples, dim=-1), uniform, reduction='batchmean')

        eps = 1e-8
        target_ratio = 0.5
        lambda_dyn = (target_ratio * (id_loss.item() + eps)) / (ood_loss.item() + eps)
        lambda_dyn = float(np.clip(lambda_dyn, 1e-4, 5.0))


        loss = id_loss + lambda_dyn * ood_loss

        return loss

    def validation_performance(self, model, loader, *args, **kwargs):
        running_loss = 0.0
        scores_dict = {}
        for batch in loader:
            b, y = self.batch_extractor(batch, model=model, **kwargs)
            output = model(b)
            out_samples = output[-self.n_adv_imgs:, :]
            uniform = torch.full_like(out_samples, 1.0 / out_samples.size(1))
            ood_loss =F.kl_div(F.log_softmax(out_samples, dim=-1), uniform, reduction='batchmean')
            running_loss += ood_loss
        scores_dict['score'] = running_loss.item() / len(loader)
        return scores_dict


    def test_performance(self, model, loader, pbar, *args, **kwargs):
        pass
