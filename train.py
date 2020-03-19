# -*- coding: utf-8 -*-
#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Runs a model on a single node across N-gpus.
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch

from lightning_module import LightningTemplateModel
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import re
import pathlib
import collections

def getLastCP():    
    py = pathlib.Path().glob("checkpoints/*.ckpt")
    cpts={}
    for file in py:
        ind=re.match('.*?_([0-9]+).*$', str(file) ).group(1)
        cpts[int(ind)]= str(file)
        print(file)
        print(ind)
    cpts = collections.OrderedDict(sorted(cpts.items()))
    k=list(cpts.keys() )
    klast=k[len(k)-1]
    print(cpts[klast])
    return cpts[klast]


#SEED = 2334
#torch.manual_seed(SEED)
#np.random.seed(SEED)
checkpoint_callback = ModelCheckpoint(
    save_top_k=-1,
    period=1,
    filepath='./checkpoints/',    
    prefix='',
    verbose=True
)

earlystopp_callback=EarlyStopping(
    monitor='val_loss',
    min_delta=0.0,
    patience=100,
    verbose=0,
    mode='auto')

logger = TensorBoardLogger(
                        save_dir=os.getcwd(),
                        name='lightning_logs'
                    )

def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningTemplateModel(hparams)
    
    last_cp= getLastCP()
    #if (last_cp!=None):        
    #    model.load_from_checkpoint(checkpoint_path=last_cp)
    
    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
   
   
    trainer = Trainer(
        default_save_path='./checkpoints/',
        logger=logger,
        amp_level='O2',
        resume_from_checkpoint=last_cp,
        use_amp=False,
        gradient_clip_val=1,
        gpus=hparams.gpus,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=earlystopp_callback,
        distributed_backend=hparams.distributed_backend,
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='how many gpus'
    )
    parent_parser.add_argument(
        '--distributed_backend',
        type=str,
        default=None,
        help='supports three options dp, ddp, ddp2'
    )
    parent_parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )

    # each LightningModule defines arguments relevant to it
    parser = LightningTemplateModel.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()
    
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
