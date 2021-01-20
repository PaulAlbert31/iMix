import torch
import torchvision
import torchvision.transforms as transforms
import os
import datasets
import torch.nn.functional as F
import numpy as np

def multi_class_loss(pred, target):
    pred = F.log_softmax(pred, dim=1)
    loss = - torch.sum(target*pred, dim=1)
    return loss

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
    else:
        lam = 1

    device = x.get_device()
    batch_size = x.size()[0]

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index


def make_data_loader(args, no_aug=False, transform=None, **kwargs):
    
    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif args.dataset == 'miniimagenet':
        mean = [0.4728, 0.4487, 0.4031]
        std = [0.2744, 0.2663 , 0.2806]

    size = 32
    if args.dataset == 'miniimagenet':
        size = 84


    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.CenterCrop(size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])


    if args.dataset == "cifar10":
        trainset = datasets.cifar10(transform=transform_train, regim='train')
        testset = datasets.cifar10(transform=transform_test, regim='val')
    elif args.dataset == "cifar100":
        trainset = datasets.cifar100(transform=transform_train, regim='train')
        testset = datasets.cifar100(transform=transform_test, regim='val')
    elif args.dataset == "miniimagenet":
        trainset, testset = datasets.miniimagenet(transform=transform_train, transform_test=transform_test)
    else:
        raise NotImplementedError
    
    if no_aug:
        train_loader =  torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, **kwargs) #Sequential loader for sample loss tracking
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs) #Normal training
        
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
        
    return train_loader, test_loader

def create_save_folder(args):
    try:
        os.mkdir(args.save_dir)
    except:
        pass
    try:
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset))
    except:
        pass
    try:
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name))
    except:
        pass
    return
       
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor