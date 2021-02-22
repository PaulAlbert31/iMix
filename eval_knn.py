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
elif net == "resnet50":
    from nets.resnet import ResNet50
    model = ResNet50(proj_size)
elif net == "wide282":
    from nets.wideresnet import WRN28_2
    model = WRN28_2(proj_size)
else:
    raise NotImplementedError

print("Number of parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

model = nn.DataParallel(model).cuda()
weights = torch.load(weights)
if "net" in weights.keys():
    weights = weights["net"]
if "network" in weights.keys():
    weights = weights["network"]
    
model.load_state_dict(weights, strict=True)

model.eval()

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

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size),
    torchvision.transforms.CenterCrop(size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])

kwargs = {"num_workers": 12, "pin_memory": False}
if dataset == "cifar10":
    trainset = datasets.cifar10(transform=transform, regim='train')
    testset = datasets.cifar10(transform=transform, regim='val')
elif dataset == "cifar100":
    trainset = datasets.cifar100(transform=transform, regim='train')
    testset = datasets.cifar100(transform=transform, regim='val')
elif dataset == "miniimagenet":
    trainset, testset = datasets.miniimagenet(transform=transform, transform_test=transform)
else:
    raise NotImplementedError

train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, **kwargs)

# tracking variables                                                                                                                                                                                                                                                      
total = 0
train_feats = torch.zeros((len(train_loader.dataset), proj_size))

with torch.no_grad():
    tbar = tqdm(train_loader)
    tbar.set_description("Computing features on the train set")
    for i, sample in enumerate(tbar):
        images, index = sample["image1"].cuda(), sample["index"]
        # forward                                                                                                                                                                                                                                                         
        features = model(images)
        train_feats[index] = features.detach().cpu()


trainFeatures = train_feats.cuda()
trainLabels = torch.LongTensor(train_loader.dataset.targets).cuda()

trainFeatures = trainFeatures.t()
C = trainLabels.max() + 1
C = C.item()
# start to evaluate                                                                                                                                                                                                                                                       
top1 = 0.
top5 = 0.
tbar = tqdm(test_loader)
tbar.set_description("kNN eval")
K = 200
sigma = 0.1
with torch.no_grad():
    retrieval_one_hot = torch.zeros(K, C).cuda()
    for i, sample in enumerate(tbar):
        images, targets = sample["image"].cuda(), sample["target"].cuda()
        batchSize = images.size(0)
        # forward                                                                                                                                                                                                                                                         
        features = model(images)
        
        # cosine similarity                                                                                                                                                                                                                                               
        dist = torch.mm(features, trainFeatures)
        
        yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
        candidates = trainLabels.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        
        retrieval_one_hot.resize_(batchSize * K, C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(sigma).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C),
                                    yd_transform.view(batchSize, -1, 1)), 1)
        _, predictions = probs.sort(1, True)
        
        # Find which predictions match the target                                                                                                                                                                                                                        
        correct = predictions.eq(targets.data.view(-1,1))
        
        top1 = top1 + correct.narrow(1,0,1).sum().item()
        top5 = top5 + correct.narrow(1,0,5).sum().item()
        
        total += targets.size(0)

    print("kNN accuracy", top1/total*100)
    print(top1, total)
