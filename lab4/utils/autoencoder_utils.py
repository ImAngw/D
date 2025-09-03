from torch import nn

from my_custom_ai.utils.misc_utils import Config
from my_custom_ai.utils.train_utils import FunctionContainer




class AutoencoderConfigs(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class AutoencoderContainer(FunctionContainer):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.loss_func = nn.MSELoss()

    def batch_extractor(self, batch, *args, **kwargs):
        imgs, _ = batch
        return imgs.to(self.device), imgs.to(self.device)

    def loss_function(self, model_output, y, *args, **kwargs):
        loss = self.loss_func(model_output, y)
        return loss

    def validation_performance(self, model, loader, *args, **kwargs):
        scores = {}
        mse = 0.

        for batch in loader:
            imgs, y = self.batch_extractor(batch)
            output = model(imgs)
            loss = self.loss_function(output, y)
            mse += loss.item()

        scores['score'] = mse / len(loader)
        return scores

    def test_performance(self, model, loader, pbar, *args, **kwargs):
        pass
