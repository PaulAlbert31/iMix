import torch
import torch.nn as nn
import torchvision
import numpy as np
import sys
from tqdm import tqdm
import datasets

proj_size = 128

weights = sys.argv[1]
s = weights.split('/')[0]

dataset = s.split('_')[0].lower()
net = s.split('_')[1].lower()

if net == "resnet18":
    from nets.resnet import ResNet18
    model = ResNet18(proj_size)
    np = 512
elif net == "resnet50":
    from nets.resnet import ResNet50
    model = ResNet50(proj_size)
    np = 2048
elif net == "wide282":
    from nets.wideresnet import WRN28_2
    model = WRN28_2(proj_size)
    np = 128
else:
    raise NotImplementedError

print("Number of parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

model = nn.DataParallel(model)
weights = torch.load(weights)
if "net" in weights.keys():
    weights = weights["net"]
if "network" in weights.keys():
    weights = weights["network"]
    
model.load_state_dict(weights, strict=True)

if dataset == 'cifar10':
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
elif dataset == 'cifar100':
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
elif dataset == 'miniimagenet':
    mean = [0.4728, 0.4487, 0.4031]
    std = [0.2744, 0.2663 , 0.2806]

size = 32
if dataset == 'miniimagenet':
    size = 84

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(size),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    torchvision.transforms.RandomGrayscale(p=0.2),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])
    
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size),
    torchvision.transforms.CenterCrop(size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])

kwargs = {"num_workers": 12, "pin_memory": False}
if dataset == "cifar10":
    trainset = datasets.cifar10(transform=transform_train, regim='train')
    testset = datasets.cifar10(transform=transform, regim='val')
    nclass = 10
elif dataset == "cifar100":
    trainset = datasets.cifar100(transform=transform_train, regim='train')
    testset = datasets.cifar100(transform=transform, regim='val')
    nclass = 100
elif dataset == "miniimagenet":
    trainset, testset = datasets.miniimagenet(transform=transform_train, transform_test=transform)
    nclass = 100
else:
    raise NotImplementedError

train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, **kwargs)

model.module.linear = nn.Linear(np, nclass)
model.cuda()

epochs = 100
best = 0
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,80], gamma=0.1)

for eps in range(epochs):
    model.train()

    #Finetune last layer
    lim = len(list(model.module.children()))
    ct = 0
    for child in model.module.children():
        ct += 1
        if ct < lim:
            for param in child.parameters():
                param.requires_grad = False

    tbar = tqdm(train_loader)
    tbar.set_description("Train {}/{}".format(eps, epochs))
    acc = 0
    total = 0
    for i, sample in enumerate(tbar):
        images, target, index = sample["image1"].cuda(), sample["target"].cuda(), sample["index"]
        # forward                                                                                                                                                                                                                                                         
        outputs = model(images)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        preds = torch.argmax(nn.functional.log_softmax(outputs, dim=1), dim=1)        
        acc += torch.sum(preds == target.data)
        total += preds.size(0)
        
        tbar.set_description("Train {}/{}, loss {:.3f}, lr {:.3f}".format(eps, epochs, loss.item(), optimizer.param_groups[0]['lr']))
    scheduler.step()
    print("Train accuracy {:.4f}".format(100.*acc/total))
    
    model.eval()
    acc = 0
    total = 0
    tbar = tqdm(test_loader)
    tbar.set_description("Test")
    for i, sample in enumerate(tbar):
        images, target, index = sample["image"].cuda(), sample["target"].cuda(), sample["index"]
        # forward                                                                                                                                                                                                                                                         
        outputs = model(images)
        loss = criterion(outputs, target)
        
        preds = torch.argmax(nn.functional.log_softmax(outputs, dim=1), dim=1)        
        acc += torch.sum(preds == target.data)
        total += preds.size(0)
        
        tbar.set_description("Test loss {:.3f}".format(loss.item()))
    acc = 1.*acc/total
    if acc > best:
        best = acc
    print("Test accuracy {:.4f}, Best {:.4f}".format(acc*100, best*100))
