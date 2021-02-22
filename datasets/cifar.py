##########################
#  CUSTOM CIFAR DATASETS #
##########################

import torchvision
import numpy as np
from PIL import Image

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, reduced=None, rotations=False):
        super(CIFAR10, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.train = train
        
    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]
        
        img = Image.fromarray(img)

        if self.train:
            return {'image1':self.transform(img), 'image2': self.transform(img), 'index':index}
        
        return {'image':self.transform(img), 'target': target, 'index':index}
            

    
class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, reduced=None, rotations=False):
        super(CIFAR100, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.train = train

    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]
                
        img = Image.fromarray(img)
            
        if self.train:
            return {'image1':self.transform(img), 'image2': self.transform(img), 'index':index}
        
        return {'image':self.transform(img), 'target': target, 'index':index}
