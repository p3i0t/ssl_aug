import os
import hydra
from omegaconf import DictConfig
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18, resnet34
from models import SimCLR
from utils import AverageMeter

from tqdm import tqdm


logger = logging.getLogger(__name__)


class LinModel(nn.Module):
    """Linear wrapper of encoder."""
    def __init__(self, encoder: nn.Module, feature_dim: int, n_classes: int):
        super().__init__()
        self.enc = encoder
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.lin = nn.Linear(self.feature_dim, self.n_classes)

    def forward(self, x):
        return self.lin(self.enc(x))


def run_epoch(model, dataloader, epoch=0, optimizer=None, scheduler=None):
    """Run one epoch, train or eval."""
    if optimizer:
        model.train()
    else:
        model.eval()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    loader_bar = tqdm(dataloader)
    for x, y in loader_bar:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        acc = (logits.argmax(dim=1) == y).float().mean()
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc.item(), x.size(0))
        if optimizer:
            loader_bar.set_description("Train epoch {}, loss: {:.4f}, acc: {:.4f}"
                                       .format(epoch, loss_meter.avg, acc_meter.avg))
        else:
            loader_bar.set_description("Test epoch {}, loss: {:.4f}, acc: {:.4f}"
                                       .format(epoch, loss_meter.avg, acc_meter.avg))

    return loss_meter.avg, acc_meter.avg


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


class CorruptionDataset(Dataset):
    # for cifar10 and cifar100
    def __init__(self, x, y, transform=None):
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, item):
        sample = self.x[item]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.y[item]

    def __len__(self):
        return self.x.shape[0]


def eval_c(classifier, base_path):
    """Evaluate network on given corrupted dataset."""
    corruption_acc_dict = {}
    acc_meter = AverageMeter('Acc')
    for corruption in CORRUPTIONS:
        # Reference to original data is mutated
        preprocess = transforms.ToTensor()

        x = np.load(base_path + corruption + '.npy')
        y = np.load(base_path + 'labels.npy').astype(np.int64)
        dataset = CorruptionDataset(x, y, transform=preprocess)

        test_loader = DataLoader(
            dataset,
            batch_size=200,
            shuffle=False,
            pin_memory=True)

        test_loss, test_acc = run_epoch(classifier, test_loader)
        corruption_acc_dict[corruption] = test_acc
        acc_meter.update(test_acc)

    return corruption_acc_dict, acc_meter.avg


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


@hydra.main(config_path='simclr_config.yml')
def finetune(args: DictConfig) -> None:
    train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor()])
    test_transform = transforms.ToTensor()

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_set = CIFAR10(root=data_dir, train=True, transform=train_transform, download=False)
    test_set = CIFAR10(root=data_dir, train=False, transform=test_transform, download=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Prepare model
    base_encoder = eval(args.backbone)
    pre_model = SimCLR(base_encoder, projection_dim=args.projection_dim).cuda()
    state_dict = torch.load('simclr_{}_epoch{}{}.pt'.format(args.backbone, args.load_epoch, '_aug' if args.aug else ''))
    pre_model.load_state_dict(state_dict)
    model = LinModel(pre_model.enc, feature_dim=pre_model.feature_dim, n_classes=len(train_set.targets))
    model = model.cuda()

    # Fix encoder
    model.enc.requires_grad = False
    parameters = [param for param in model.parameters() if param.requires_grad is True]  # trainable parameters.
    # optimizer = Adam(parameters, lr=0.001)

    if args.eval:
        save_path = 'simclr_lin_{}{}_best.pth'.format(args.backbone, '_aug' if args.aug else '')
        model.load_state_dict(torch.load(save_path))

        train_loss, train_acc = run_epoch(model, train_loader)
        test_loss, test_acc = run_epoch(model, test_loader)

        logger.info("Evaluation on {}".format(args.dataset))
        logger.info("Train Acc: {:.4f}".format(train_acc))
        logger.info("Test Acc: {:.4f}".format(test_acc))

        if args.dataset == 'cifar10':
            base_c_path = os.path.join(data_dir, 'CIFAR-10-C/')
        else:
            base_c_path = os.path.join(data_dir, 'CIFAR-100-C/')

        corruption_acc_dict, corruption_acc = eval_c(model, base_c_path)
        logger.info("Mean Acc on Corrupted Dataset: {:.4f}".format(corruption_acc))
        torch.save(corruption_acc_dict, 'corruption_results_{}.pth'.format(args.dataset))
    else:
        optimizer = torch.optim.SGD(
            parameters,
            0.2,   # lr = 0.1 * batch_size / 256, see section B.6 and B.7 of SimCLR paper.
            momentum=args.momentum,
            weight_decay=0.,
            nesterov=True)

        # cosine annealing lr
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
                step,
                args.epochs * len(train_loader),
                args.learning_rate,  # lr_lambda computes multiplicative factor
                1e-3))

        optimal_loss, optimal_acc = 1e5, 0.
        for epoch in range(1, args.finetune_epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, epoch, optimizer, scheduler)
            test_loss, test_acc = run_epoch(model, test_loader, epoch)

            if train_loss < optimal_loss:
                optimal_loss = train_loss
                optimal_acc = test_acc
                logger.info("==> New optimal test acc: {:.4f} found.".format(optimal_acc))
                save_path = 'simclr_lin_{}{}_best.pth'.format(args.backbone, '_aug' if args.aug else '')
                torch.save(model.state_dict(), save_path)

        logger.info("Best Test Acc: {:.4f}".format(optimal_acc))


if __name__ == '__main__':
    finetune()


