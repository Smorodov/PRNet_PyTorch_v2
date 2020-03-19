# -*- coding: utf-8 -*-
"""
Example template for defining a system
"""
import logging as log
from argparse import ArgumentParser
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from torchvision.utils import make_grid 

from resfcn256 import ResFCN256
from WLP300dataset import PRNetDataset, ToTensor, ToNormalize, RescaleAndCrop, FlipH

#import pytorch_msssim
from losses import SSIM

import win32com.client as wincl
speak = wincl.Dispatch("SAPI.SpVoice")
# speak.Speak("")
from cv_plot import plot_kpt
import cv2
import numpy as np

uv_kpt_ind_path = "uv_data/uv_kpt_ind.txt"
face_ind_path =  "uv_data/face_ind.txt"
triangles_path = "uv_data/triangles.txt"

uv_kpt_ind = np.loadtxt(uv_kpt_ind_path).astype(np.int32)  # 2 x 68 get kpt
face_ind = np.loadtxt(face_ind_path).astype(np.int32)  # get valid vertices in the pos map
triangles = np.loadtxt(triangles_path).astype(np.int32)  # ntri x 3

weights_img=cv2.imread('uv_data/uv_weight_mask_gdh.png')
mask_image = np.zeros(shape=[256, 256, 3], dtype=np.float32)

for i in range(256*256):
    x=i//256
    y=i%256
    if weights_img[y,x,:].any()>0:        
        mask_image[y,x,:]=1.0

#cv2.imshow("mask",mask_image)
#cv2.waitKey()

def generate_uv_coords():
    resolution = 256
    uv_coords = np.meshgrid(range(resolution), range(resolution))
    # uv_coords = np.transpose(np.array(uv_coords), [1, 2, 0])
    uv_coords = np.reshape(uv_coords, [resolution ** 2, -1])
    uv_coords = uv_coords[face_ind, :]
    uv_coords = np.hstack((uv_coords[:, :2], np.zeros([uv_coords.shape[0], 1])))
    return uv_coords

def get_landmarks(pos):
    '''
    Notice: original tensorflow version shape is [256, 256, 3] (H, W, C)
            where our pytorch shape is [3, 256, 256] (C, H, W).

    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        kpt: 68 3D landmarks. shape = (68, 3).
    '''
    
    kpt = pos[uv_kpt_ind[1, :], uv_kpt_ind[0, :], :]
    return kpt

def get_vertices(pos):
    '''
    Args:
        pos: the 3D position map. shape = (3, 256, 256).
    Returns:
        vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
    '''
    all_vertices = np.reshape(pos, [256 ** 2, -1])
    vertices = all_vertices[face_ind, :]
    return vertices

uv_coords = generate_uv_coords()
# -------------------------
# 
# -------------------------  
def show_batch(img, pos):    
    h,w = img.shape[2],img.shape[3]
    batch_size=img.shape[0]
    result=torch.Tensor(batch_size,3,h,w)    
    img_cpu = img.to("cpu").detach().numpy()*255
    pos_cpu = pos.to("cpu").detach().numpy()*255    

    for i in range(min(img_cpu.shape[0],pos_cpu.shape[0])):
        im = img_cpu[i,:,:,:]
        im = im.squeeze()
        im = im.transpose((1,2,0))         
        
        pos = pos_cpu[i,:,:,:]
        pos = pos.squeeze()        
        
        pos = pos.transpose(1, 2, 0)        
        kpt = get_landmarks(pos)
        im = plot_kpt(im, kpt)
        #im=plot_landmarks(im, (annotation_cpu[i])*224)              

        im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        im = torch.from_numpy(im).float()
        
        im = im[np.newaxis, :]     
        im = im.permute(0, 3, 1, 2)        
        result[i,:,:,:]=im       
    return result

# -------------------------
# 
# -------------------------  
def show_res(im, pos):                     
    kpt = get_landmarks(pos)
    im = plot_kpt(im, kpt)        
    return im


class LightningTemplateModel(LightningModule):
    """
    Sample model to show how to define a template
    """
   
    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """

        # init superclass
        super(LightningTemplateModel, self).__init__()
        self.hparams = hparams
                
        
        #self.trainer = pl.Trainer(logger=self.logger, accumulate_grad_batches=2,amp_level='O2', use_amp=False)
        #self.trainer = pl.Trainer(default_save_path='./checkpoints/', logger=self.logger, amp_level='O2', use_amp=False, checkpoint_callback=checkpoint_callback)
        self.batch_size = hparams.batch_size

        #if you specify an example input, the summary will show input/output for each layer
        # self.example_input_array = torch.rand(64, 3, 224, 224)

        # build model
        self.__build_model()
        
        self.ssim_loss = SSIM(mask_path="uv_data/uv_weight_mask_gdh.png", gauss="original")
        
        #self.ssim_loss = pytorch_msssim.SSIM(size_average=False)

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout model
        :return:
        """        
        self.model = ResFCN256()
          
    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        x=self.model.forward(x)
        return x

    def loss(self, targets, outputs):
        ssim = self.ssim_loss(targets, outputs)        
        return ssim

    def on_epoch_end(self):
        # do something when the epoch ends 
        torch.save(self.model,'model_converter/model.pth')
        speak.Speak("Эпоха окончена")

    def on_epoch_start(self):
        speak.Speak("Эпоха начата")

        
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # self.train(True)
        # forward pass
        x, y = batch['origin'],batch['uv_map']
        y_hat = self.forward(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)
        
        
        if (self.global_step % 500) == 0:                        
            # self.logger.experiment.add_image('train_results',make_grid(y_hat), batch_idx)    
            map_gt, map_pred = make_grid(y), make_grid(y_hat)            
            gr=make_grid(show_batch(x, y_hat),normalize=True)
            gr_gt=make_grid(show_batch(x, y),normalize=True)
            
            self.logger.experiment.add_image('map_gt', map_gt, batch_idx)
            self.logger.experiment.add_image('map_pred', map_pred, batch_idx)
            self.logger.experiment.add_image('gt', gr_gt, batch_idx)
            self.logger.experiment.add_image('pred', gr, batch_idx)                
        
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        if (self.global_step % 500) == 0:            
            speak.Speak("Итерация "+str(self.global_step))
            speak.Speak('Лосс {:.3f}'.format(loss_val))


        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output
        
    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        # self.train(False)
        
        x, y = batch['origin'],batch['uv_map']
        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)
        
        
        if (batch_idx  == 0):                        
            #self.logger.experiment.add_image('val_results',make_grid(y_hat), batch_idx)    
            map_gt, map_pred = make_grid(y), make_grid(y_hat)            
            gr=make_grid(show_batch(x, y_hat),normalize=True)
            gr_gt=make_grid(show_batch(x, y),normalize=True)
            
            self.logger.experiment.add_image('val_map_gt', map_gt, batch_idx)
            self.logger.experiment.add_image('val_map_pred', map_pred, batch_idx)
            self.logger.experiment.add_image('val_gt', gr_gt, batch_idx)
            self.logger.experiment.add_image('val_pred', gr, batch_idx)                


        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)            
            
        output = OrderedDict({
            'val_loss': loss_val,
        })
        
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

        val_loss_mean /= len(outputs)        
        speak.Speak('Тест Лосс {:.3f}'.format(val_loss_mean))
        tqdm_dict = {'val_loss': val_loss_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        
        # img=show_landmarks_batch(output['x'],output['y_hat'])
        #self.logger.experiment.add_image('val_results',make_grid(img), 0)    
        
        
        return result
               
    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate,betas=(0.5, 0.999))       
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        
        return [optimizer], [scheduler]

    def __dataloader(self, train):

        data_dir ='Y:/PRNet_PyTorch/utils/300WLP_IBUG'        
        dataset=PRNetDataset(root_dir=data_dir,
                             train=train,
                             transform=transforms.Compose([ FlipH(),RescaleAndCrop(),                                                         
                                                         ToTensor(),                                                        
                                                         ToNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                         ]))      
        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = None
        batch_size = self.hparams.batch_size

        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)

        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=dataset,
            drop_last=True,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=2
        )

        return loader

    @pl.data_loader
    def train_dataloader(self):
        log.info('Training data loader called.')
        return self.__dataloader(train=True)

    @pl.data_loader
    def val_dataloader(self):
        log.info('Validation data loader called.')
        return self.__dataloader(train=False)

    @pl.data_loader
    def test_dataloader(self):
        log.info('Test data loader called.')
        return self.__dataloader(train=False)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        # use 500 for CPU, 50000 for GPU to see speed difference
        parser.add_argument('--learning_rate', default=0.0001, type=float)
        # training params (opt)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--batch_size', default=48, type=int)
        return parser
