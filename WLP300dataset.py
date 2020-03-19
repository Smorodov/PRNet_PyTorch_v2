# -*- coding: utf-8 -*-
"""
    @author: samuel ko
    @date: 2019.07.18
    @readme: The implementation of PRNet Network DataLoader.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

import cv2
from glob import glob
import random
import numbers
import numpy as np
from PIL import Image
from skimage import io
from skimage import io, transform


weights_img=cv2.imread('uv_data/uv_weight_mask_gdh.png').astype(np.float32)
mask_image = np.zeros(shape=[256, 256, 3], dtype=np.float32)

for i in range(256*256):
    x=i//256
    y=i%256
    if weights_img[y,x,:].any()>0:        
        mask_image[y,x,:]=1.0

#mask_image = torch.from_numpy(mask_image.transpose((2, 0, 1)))        


'''
data_transform = {'train': transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}
'''

class PRNetDataset(Dataset):
    """Pedestrian Attribute Landmarks dataset."""

    def __init__(self, root_dir,train, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.train=train
        self.n_total = int(len(os.listdir(self.root_dir)))
        self.n_train = int(self.n_total*0.8)
        self.n_test = int(self.n_total-self.n_train)
        # shuffle the indexes
        self.indexes = np.arange(self.n_total-1)        
        # use 'n_train' samples for training and the rest for testing
        self.train_ids = self.indexes[:self.n_train]
        self.test_ids = self.indexes[-self.n_test:]

    def get_img_path(self, img_id):
    #img_id = self.dict.get(img_id)        
        original = os.path.join(self.root_dir, str(img_id), 'original.jpg')
        # fixme: Thanks to mj, who fix an important bug!
        uv_map_path = glob(os.path.join(self.root_dir, str(img_id), "*.npy"))
        uv_map = uv_map_path[0]

        return original, uv_map
    '''
    def _max_idx(self):
        _tmp_lst = map(lambda x: int(x), os.listdir(self.root_dir))
        _sorted_lst = sorted(_tmp_lst)
        for idx, item in enumerate(_sorted_lst):
            self.dict[idx] = item
    '''
    def __len__(self):
        if(self.train):
            return self.n_train
        else:
            return self.n_test

    def __getitem__(self, idx):
        while(1):
            idx=0
            if(self.train):
                idxidx=random.randint(0,self.n_train-1)            
                idx=self.train_ids[idxidx]
            else:
                idxidx=random.randint(0,self.n_test-1)            
                idx=self.test_ids[idxidx]
                
            #idx=0
            try:            
                original, uv_map = self.get_img_path(idx)    
                origin = cv2.imread(original)
                uv_map = np.load(uv_map)
                break
            except:
                idx=random.randint(0,self.__len__()-1)

        sample = {'uv_map': uv_map, 'origin': origin}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        uv_map, origin = sample['uv_map'], sample['origin']
        
        # uv_map=cv2.cvtColor(uv_map,cv2.COLOR_BGR2RGB)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        uv_map = uv_map.transpose((2, 0, 1))
        origin = origin.transpose((2, 0, 1))

        uv_map = uv_map.astype("float32") / 255.
        uv_map = np.clip(uv_map, 0, 1)
        origin = origin.astype("float32")
        # origin = origin.astype("float32") / 255.
        
        
        return {'uv_map': torch.from_numpy(uv_map), 'origin': torch.from_numpy(origin)}


class ToNormalize(object):
    """Normalized process on origin Tensors."""

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        uv_map, origin = sample['uv_map'], sample['origin']
        origin = F.normalize(origin, self.mean, self.std, self.inplace)
        return {'uv_map': uv_map, 'origin': origin}


class RescaleAndCrop(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __call__(self, sample):                        
        uv_map, origin = sample['uv_map'], sample['origin']
        try:            
            h, w = origin.shape[:2]
            scaled_output_size = random.randint(int(w),int(w*1.25) )
            
            new_h = scaled_output_size+1
            new_w = scaled_output_size+1
            new_h, new_w = int(new_h), int(new_w)
            img = transform.resize(origin, (new_h, new_w))
            sx=new_w / w
            sy=new_h / h        
            uv_map[:,:,0]=uv_map[:,:,0]*sx
            uv_map[:,:,1]=uv_map[:,:,1]*sy
    
            h=new_h
            w=new_w
            new_h = 256
            new_w = 256
    
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
    
            image = img[top: top + new_h,
                          left: left + new_w]
    
            
            uv_map[:,:,0]-=left
            uv_map[:,:,1]-=top
            uv_map*=mask_image            
        except:
            image=origin.copy()
        return {'uv_map': uv_map, 'origin': image}

class FlipH(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __call__(self, sample):                        
        uv_map, origin = sample['uv_map'], sample['origin']
        P = random.randint(0,100 )
        if(P>59):
            
            try:            
                h, w = origin.shape[:2]
                img=cv2.flip(img, 1)
                uv_map=cv2.flip(uv_map, 1)
                
                #uv_map[:,:,0]=uv_map[:,:,0]
                uv_map[:,:,0]=255-uv_map[:,:,0]                         
            except:
                img=origin.copy()
        else:
            img=origin.copy()
        return {'uv_map': uv_map, 'origin': img}
       
