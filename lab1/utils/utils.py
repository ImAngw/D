import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from my_custom_ai.utils.misc_utils import Config
from my_custom_ai.utils.train_utils import FunctionContainer


import torchvision.transforms as transforms
import torchvision
import torch


from tqdm import tqdm
import sys



class Configs(Config):
    def __init__(
            self,
            depth: int,
            lr: float,
            dropout: float,
            dim_hidden_layers: list =None,      # used in mlp models
            channels_hidden_conv: list = None,  # used in cnn models
            img_reductions: list = None,        # used in cnn models
            expansion_factor: int = 1,
            require_skip_connection: bool = False,
            block_type: str = 'mlp',    # allowed: mpl, cnn
            dataset: str = 'mnist',     # allowed: mnist, cifar10

            save_on_wb=False,
            **kwargs
    ):
        self.depth = depth
        self.lr = lr
        self.dropout = dropout
        self.require_skip_connection = require_skip_connection
        self.block_type = block_type
        self.dataset = dataset
        self.img_reductions = img_reductions
        self.expansion_factor = expansion_factor
        # list with a structure as [(in_dim0, out_dim0), ..., (in_dimN, out_dimN)] with N=depth
        self.dim_hidden_layers = dim_hidden_layers
        self.channels_hidden_conv = channels_hidden_conv

        logger_init = {
            'entity': "your_name",
            'project': 'your_project_name',
            'name': kwargs['experiment_name'],
            'configs': {
                'dataset': dataset,
                'n_blocks': depth,
                'skip_connection': require_skip_connection
            }
        }

        if save_on_wb:
            super().__init__(logger_init=logger_init, **kwargs)
        else:
            super().__init__(**kwargs)



class ClassificationContainer(FunctionContainer):
    def __init__(self, is_mlp, device='cpu'):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.is_mlp = is_mlp
        self.device = device


    def batch_extractor(self, batch, *args, **kwargs):
        b, y = batch
        if self.is_mlp:
            b = torch.flatten(b, start_dim=1)
        return b.to(self.device), y.to(self.device)

    def loss_function(self, model_output, y, *args, **kwargs):
        loss = self.criterion(model_output, y)
        return loss


    def validation_performance(self, model, loader, *args, **kwargs):
        total = 0
        corrects = 0
        loss = 0.

        scores_dict = {}

        for batch in loader:
            b, y = self.batch_extractor(batch, *args, **kwargs)
            output = model(b)

            predictions = torch.argmax(output, dim=-1)
            correct = torch.sum(predictions == y)

            loss += self.criterion(output, y).item()

            total += y.size(0)
            corrects += correct.item()

        # score = corrects / total
        scores_dict['score'] = corrects / total
        scores_dict['val_loss'] = loss / len(loader)

        return scores_dict


    def test_performance(self, model, loader, *args, **kwargs):
        return self.validation_performance(model, loader, *args, **kwargs)


class DistillationContainer(FunctionContainer):
    def __init__(self, device='cpu'):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.kl_divergence = nn.KLDivLoss(reduction='batchmean')
        self.device = device


    def batch_extractor(self, batch, *args, **kwargs):
        b, label, probs = batch
        return b.to(self.device), {'label': label.to(self.device), 'probs': probs.to(self.device)}

    def loss_function(self, model_output, y, *args, **kwargs):
        alpha = 0.5
        T = 1.25

        hard_loss = self.criterion(model_output, y['label'])

        stud_log_probs = F.log_softmax(model_output / T, dim=-1)
        teacher_probs = F.softmax(y['probs'] / T, dim=-1)
        soft_loss = self.kl_divergence(stud_log_probs, teacher_probs)

        loss = alpha * soft_loss + (1 - alpha) * hard_loss
        return loss


    def validation_performance(self, model, loader, *args, **kwargs):
        total = 0
        corrects = 0
        loss = 0.

        scores_dict = {}

        for batch in loader:
            b, y = self.batch_extractor(batch, *args, **kwargs)
            output = model(b)

            predictions = torch.argmax(output, dim=-1)
            correct = torch.sum(predictions == y['label'])

            loss += self.loss_function(output, y).item()

            total += y['label'].size(0)
            corrects += correct.item()

        # score = corrects / total
        scores_dict['score'] = corrects / total
        scores_dict['val_loss'] = loss / len(loader)

        return scores_dict


    def test_performance(self, model, loader, *args, **kwargs):
        pass



def return_loaders(batch_size, dataset, distillation=False):

    mean = 0.2860 if dataset == 'mnist' else (0.4914, 0.4822, 0.4465)
    std = 0.3205 if dataset == 'mnist' else (0.2470, 0.2435, 0.2616)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)



    if distillation:
        train_data = torch.load("../lab1/augmented_datasets/train_cifar_distill.pth")
        train_probs = ReturnProbsForDistillation.__new__(ReturnProbsForDistillation)
        train_probs.probs = train_data["probs"]

        test_data = torch.load("../lab1/augmented_datasets/val_cifar_distill.pth")
        test_probs = ReturnProbsForDistillation.__new__(ReturnProbsForDistillation)
        test_probs.probs = test_data["probs"]

        train_dataset = ReturnDatasetForDistillation(train_dataset, train_probs)
        test_dataset = ReturnDatasetForDistillation(test_dataset, test_probs)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class ReturnProbsForDistillation(Dataset):
    def __init__(self, original_dataset, teacher_model):
        self.probs = []

        with tqdm(total=len(original_dataset), desc=f'BUILDING', file=sys.stdout, colour='blue', ncols=100, dynamic_ncols=True) as pbar:
            teacher_model.eval()
            with torch.no_grad():
                for img, y in original_dataset:
                    prob = teacher_model(img.unsqueeze(0))
                    self.probs.append(prob)
                    pbar.update(1)

    def __len__(self):
        return len(self.probs)

    def __getitem__(self, idx):
        return  self.probs[idx]


class ReturnDatasetForDistillation(Dataset):
    def __init__(self, original_dataset, probs_dataset):
        self.original_dataset = original_dataset
        self.probs_dataset = probs_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]
        probs = self.probs_dataset[idx]
        return  img, label, probs


