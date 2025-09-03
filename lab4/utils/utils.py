from torchvision.datasets import FakeData, CIFAR10, CIFAR100
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from lab1.models.models import CNN
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import torchvision
import random


class GetModelAndLoaders:
    def __init__(self, batch_size, device, best_model_path="../lab1/checkpoints/teacher.pth"):
        self.batch_size = batch_size
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        self.best_model_path = best_model_path

        self.cifar10_classes = None

    def get_id_loaders(self):
        test_dataset = CIFAR10(root='../Lab1/data/', train=False, download=False, transform=self.transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        train_dataset = CIFAR10(root='../Lab1/data/', train=True, download=False, transform=self.transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.cifar10_classes = test_dataset.classes

        return train_loader, test_loader

    @staticmethod
    def get_random_img(loader):
        imgs = list(loader)
        idx = random.randint(0, len(loader) - 1)
        img, label = imgs[idx]
        return img, label

    def get_fake_loader(self, n_examples):
        fakeset = FakeData(size=n_examples, image_size=(3, 32, 32), transform=self.transform)
        return torch.utils.data.DataLoader(fakeset, batch_size=self.batch_size, shuffle=False)

    def get_ood_loader(self):
        dataset = CIFAR100(root='../Lab1/data/', train=True, download=False, transform=self.transform)
        all_the_classes = dataset.classes
        people_classes = ['baby', 'boy', 'girl', 'man', 'woman']
        people_indices = [all_the_classes.index(c) for c in people_classes]

        indices = [i for i, label in enumerate(dataset.targets) if label in people_indices]
        dataset = Subset(dataset, indices)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def get_best_model(self):
        model = CNN(
            dataset='cifar10',
            channels_hidden_conv=[(128, 128), (128, 512)],
            img_reductions=[16, 16],
            require_skip_connection=True,
            depth=2
        ).to(self.device)

        model.load_state_dict(torch.load(self.best_model_path))
        return model

    def get_best_model_like(self):
        model = CNN(
            dataset='cifar10',
            channels_hidden_conv=[(128, 128), (128, 512)],
            img_reductions=[16, 16],
            require_skip_connection=True,
            depth=2
        ).to(self.device)
        return model

class ComputeScores:
    def __init__(self, model, ae_model, device, id_loader, score_function):
        self.model = model
        self.ae_model = ae_model
        self.device = device
        self.func_label = score_function

        if score_function == 'MAX_LOGIT':
            self.score_function = self._max_logit
        elif score_function == 'MAX_SOFTMAX':
            self.score_function = self._max_softmax
        else:
            raise NotImplementedError

        self.id_scores = self.compute_scores(id_loader)
        self.mse_scores = self.compute_mse_scores(id_loader)

    @staticmethod
    def _max_logit(logit):
        s = logit.max(dim=1)[0]  # get the max for each element of the batch
        return s

    @staticmethod
    def _max_softmax(logit, T=1.0):
        s = F.softmax(logit / T, 1)
        s = s.max(dim=1)[0]  # get the max for each element of the batch
        return s

    def compute_scores(self, data_loader):
        scores = []
        self.model.eval()
        with torch.no_grad():
            for data in data_loader:
                x, y = data
                output = self.model(x.to(self.device))
                s = self.score_function(output)
                scores.append(s)
            scores_t = torch.cat(scores)
            return scores_t

    def compute_mse_scores(self, data_loader):
        scores = []
        self.model.eval()
        loss = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for data in data_loader:
                x, y = data
                x = x.to(self.device)
                xr = self.ae_model(x)
                l = loss(x, xr)
                score = l.mean([1, 2, 3])
                scores.append(score)
        return torch.cat(scores)

    @staticmethod
    def _plot_roc_curve(test_scores, fake_scores):

        y_pred = torch.cat((fake_scores, test_scores))
        gt = torch.cat((torch.ones_like(fake_scores), torch.zeros_like(test_scores)))

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Titolo generale
        fig.suptitle("OOD detection performance", fontsize=16)

        # --- ROC curve ---
        RocCurveDisplay.from_predictions(gt.cpu(), y_pred.cpu(),  ax=ax[0])
        ax[0].set_title("ROC Curve")

        # --- Precision-Recall curve ---
        PrecisionRecallDisplay.from_predictions(gt.cpu(), y_pred.cpu(), ax=ax[1])
        ax[1].set_title("Precision-Recall Curve")

        plt.tight_layout()
        plt.show()

    def plot_scores_curves(self, ood_loader, is_ae=False):
        if is_ae:
            ood_scores = self.compute_mse_scores(ood_loader)
            id_scores = self.mse_scores
        else:
            ood_scores = self.compute_scores(ood_loader)
            id_scores = self.id_scores

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(sorted(id_scores.cpu()), label='ID Data')
        plt.plot(sorted(ood_scores.cpu()), label='OOD Data')
        plt.legend()
        plt.title("Line Plot")
        plt.xlabel("Ordered Samples")
        plt.ylabel("Score")

        plt.subplot(1, 2, 2)
        plt.hist(id_scores.cpu(), density=True, alpha=0.5, bins=25, label='ID Data')
        plt.hist(ood_scores.cpu(), density=True, alpha=0.5, bins=25, label='OOD Data')
        plt.legend()
        plt.title("Histogram")
        plt.xlabel("Score")
        plt.ylabel("Density")


        func_label = 'MSE' if is_ae else self.func_label
        plt.suptitle(f"ID vs OOD Scores Comparison - {func_label}", fontsize=16)
        plt.tight_layout()
        plt.show()

        self._plot_roc_curve(id_scores, ood_scores)

class NormalizeInverse(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

class ImagesPrinter:
    def __init__(self, mean, std, labels):
        self.inverse_transform = NormalizeInverse(mean, std)
        self.labels = labels


    def show_image(self, ax, img, title):
        img = self.inverse_transform(img.squeeze())
        ax.imshow(img.permute(1, 2, 0).detach().cpu())
        ax.set_title(title)
        ax.axis('off')

    def __call__(self, img1, label1, img2, label2):
        diff = img2 - img1
        diff_flat = diff.flatten()

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        self.show_image(axes[0], img1, f'Original: {self.labels[label1]}')
        self.show_image(axes[1], img2, f'ADV: {self.labels[label2]}')
        self.show_image(axes[2], diff, 'Difference')

        axes[3].hist(diff_flat.detach().cpu())
        plt.show()
