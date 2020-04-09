from __future__ import print_function
import utils.functions
import models.model as models
import argparse
import os
from utils.imresize import imresize
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from models.TuiGAN import *
from options.config import get_arguments

def TuiGAN_generate(Gs,Zs,reals,NoiseAmp, Gs2,Zs2,reals2,NoiseAmp2, opt,in_s=None,gen_start_scale=0):
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    x_ab = in_s
    x_aba = in_s
    count = 0
    if opt.mode == 'train':
        dir2save = '%s/%s/gen_start_scale=%d' % (opt.out, opt.input_name, gen_start_scale)
    else:
        dir2save = functions.generate_dir2save(opt)
    try:
        os.makedirs(dir2save)
    except OSError:
        pass
    for G,G2,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Gs2,Zs,reals,reals[1:],NoiseAmp):
        z = functions.generate_noise([3, Z_opt.shape[2] , Z_opt.shape[3] ], device=opt.device)
        z = z.expand(real_curr.shape[0], 3, z.shape[2], z.shape[3])
        x_ab = x_ab[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
        z_in = noise_amp*z+real_curr
        x_ab = G(z_in.detach(),x_ab)
        
        x_aba = G2(x_ab,x_aba)
        x_ab = imresize(x_ab.detach(),1/opt.scale_factor,opt)
        x_ab = x_ab[:,:,0:real_next.shape[2],0:real_next.shape[3]]
        x_aba = imresize(x_aba.detach(),1/opt.scale_factor,opt)
        x_aba = x_aba[:,:,0:real_next.shape[2],0:real_next.shape[3]]
        count += 1
        plt.imsave('%s/x_ab_%d.png' % (dir2save,count), functions.convert_image_np(x_ab.detach()), vmin=0,vmax=1)
    return x_ab.detach()

