import utils.functions as functions
import models.model as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from utils.imresize import imresize
import itertools
from torchvision.utils import save_image
from torchvision import transforms as T
from PIL import Image
import numpy as np


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


def train(opt,Gs,Zs,reals,NoiseAmp, Gs2,Zs2,reals2,NoiseAmp2):
    real_, real_2 = functions.read_two_domains(opt)
    in_s = 0
    in_s2 = 0
    scale_num = 0
    real = imresize(real_,opt.scale1,opt)
    real2 = imresize(real_2,opt.scale1,opt)
    reals = functions.creat_reals_pyramid(real,reals,opt)
    reals2 = functions.creat_reals_pyramid(real2,reals2,opt)
    nfc_prev = 0
    
    while scale_num<opt.stop_scale+1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass

        D_curr,G_curr, D_curr2,G_curr2 = init_models(opt)
        
        if (nfc_prev==opt.nfc):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)))
            G_curr2.load_state_dict(torch.load('%s/%d/netG2.pth' % (opt.out_,scale_num-1)))
            D_curr2.load_state_dict(torch.load('%s/%d/netD2.pth' % (opt.out_,scale_num-1)))
        
        z_curr,in_s,G_curr, z_curr2,in_s2,G_curr2 = train_single_scale(D_curr,G_curr, reals,Gs,Zs,in_s,NoiseAmp, D_curr2,G_curr2, reals2,Gs2,Zs2,in_s2,NoiseAmp2, opt,scale_num)
        
        G_curr = functions.reset_grads(G_curr,False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr,False)
        D_curr.eval()
        
        G_curr2 = functions.reset_grads(G_curr2,False)
        G_curr2.eval()
        D_curr2 = functions.reset_grads(D_curr2,False)
        D_curr2.eval()
        
        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)
        
        Gs2.append(G_curr2)
        Zs2.append(z_curr2)
        NoiseAmp2.append(opt.noise_amp2)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))
        
        torch.save(Zs2, '%s/Zs2.pth' % (opt.out_))
        torch.save(Gs2, '%s/Gs2.pth' % (opt.out_))
        torch.save(reals2, '%s/reals2.pth' % (opt.out_))
        torch.save(NoiseAmp2, '%s/NoiseAmp2.pth' % (opt.out_))

        scale_num+=1
        nfc_prev = opt.nfc
        del D_curr,G_curr, D_curr2,G_curr2
    return


def train_single_scale(netD,netG,reals,Gs,Zs,in_s,NoiseAmp, netD2,netG2,reals2,Gs2,Zs2,in_s2,NoiseAmp2, opt,scale_num,centers=None):
    real = reals[len(Gs)]
    real2 = reals2[len(Gs2)]
    save_image(denorm(real.data.cpu()), '%s/real_scale.png' % (opt.outf))
    save_image(denorm(real2.data.cpu()), '%s/real_scale2.png' % (opt.outf))

    opt.bsz = real.shape[0]
    opt.nzx = real.shape[2]
    opt.nzy = real.shape[3]

    pad_noise = 0
    pad_image = 0
    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))

    lambda_idt = opt.lambda_idt
    lambda_cyc = opt.lambda_cyc
    lambda_tv = opt.lambda_tv

    z_opt = torch.full([opt.bsz, opt.nc_z, opt.nzx,opt.nzy], 0, device=opt.device)
    z_opt = m_noise(z_opt)
    z_opt2 = torch.full([opt.bsz, opt.nc_z, opt.nzx,opt.nzy], 0, device=opt.device)
    z_opt2 = m_noise(z_opt2)

    # setup optimizer
    optimizerD = optim.Adam(itertools.chain(netD.parameters(),netD2.parameters()), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(itertools.chain(netG.parameters(),netG2.parameters()), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    loss_print = {}

    for epoch in range(opt.niter):
        noise_ = functions.generate_noise([3,opt.nzx,opt.nzy], device=opt.device)
        noise_ = m_noise(noise_.expand(opt.bsz,3,opt.nzx,opt.nzy))
        
        noise_2 = functions.generate_noise([3,opt.nzx,opt.nzy], device=opt.device)
        noise_2 = m_noise(noise_2.expand(opt.bsz,3,opt.nzx,opt.nzy))

        ############################
        # (1) Update D network
        ###########################
        for j in range(opt.Dsteps):
            # D(real)
            optimizerD.zero_grad()

            output = netD(real2).to(opt.device)
            errD_real = -output.mean()
            errD_real.backward(retain_graph=True)
            loss_print['errD_real'] = errD_real.item()
            
            output2 = netD2(real).to(opt.device)
            errD_real2 = -output2.mean()
            errD_real2.backward(retain_graph=True)
            loss_print['errD_real2'] = errD_real2.item()

            if (j == 0) & (epoch == 0):
                if Gs == []:
                    prev = torch.full([opt.bsz,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    in_s = prev
                    prev = m_image(prev)
                    c_prev = torch.full([opt.bsz,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    z_prev = torch.full([opt.bsz,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    z_prev = m_noise(z_prev)
                    opt.noise_amp = opt.noise_amp
                    
                    prev2 = torch.full([opt.bsz,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    in_s2 = prev2
                    prev2 = m_image(prev2)
                    c_prev2 = torch.full([opt.bsz,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    z_prev2 = torch.full([opt.bsz,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    z_prev2 = m_noise(z_prev2)
                    opt.noise_amp2 = opt.noise_amp2
                else:
                    prev2, c_prev2 = cycle_rec(Gs2,Gs,Zs2,reals2,NoiseAmp2,in_s2,m_noise,m_image,opt,epoch)
                    prev, c_prev = cycle_rec(Gs,Gs2,Zs,reals,NoiseAmp,in_s,m_noise,m_image,opt,epoch)
                    
                    z_prev2 = draw_concat(Gs,Zs2,reals2,NoiseAmp2,in_s2,'rec',m_noise,m_image,opt)
                    z_prev2 = m_image(z_prev2)
                    z_prev = draw_concat(Gs2,Zs,reals,NoiseAmp,in_s,'rec',m_noise,m_image,opt)
                    z_prev = m_image(z_prev)
            else:
                prev2, c_prev2 = cycle_rec(Gs2,Gs,Zs2,reals2,NoiseAmp2,in_s2,m_noise,m_image,opt,epoch)
                prev, c_prev = cycle_rec(Gs,Gs2,Zs,reals,NoiseAmp,in_s,m_noise,m_image,opt,epoch)

            noise = opt.noise_amp * noise_ + m_image(real)
            noise2 = opt.noise_amp2 * noise_2 + m_image(real2)

            fake = netG(noise.detach(), prev)
            output = netD(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            loss_print['errD_fake'] = errD_fake.item()

            gradient_penalty = functions.calc_gradient_penalty(netD, real2, fake, opt.lambda_grad, opt.device)
            gradient_penalty.backward()
            loss_print['gradient_penalty'] = gradient_penalty.item()

            fake2 = netG2(noise2.detach(), prev2)
            output2 = netD2(fake2.detach())
            errD_fake2 = output2.mean()
            errD_fake2.backward(retain_graph=True)
            loss_print['errD_fake2'] = errD_fake2.item()

            gradient_penalty2 = functions.calc_gradient_penalty(netD2, real, fake2, opt.lambda_grad, opt.device)
            gradient_penalty2.backward()
            loss_print['gradient_penalty2'] = gradient_penalty2.item()

            optimizerD.step()


        ############################
        # (2) Update G network
        ###########################
        for j in range(opt.Gsteps):
            loss_tv = TVLoss()
            optimizerG.zero_grad()
            output = netD(fake)
            errG = -output.mean() + lambda_tv*loss_tv(fake)
            errG.backward(retain_graph=True)
            loss_print['errG'] = errG.item()

            output2 = netD2(fake2)
            errG2 = -output2.mean() + lambda_tv*loss_tv(fake2)
            errG2.backward(retain_graph=True)
            loss_print['errG2'] = errG2.item()

            loss = nn.L1Loss()
            Z_opt2 =  m_image(real2)
            rec_loss = lambda_idt*loss(netG(Z_opt2.detach(),z_prev2),real2)
            rec_loss.backward(retain_graph=True)
            loss_print['rec_loss'] = rec_loss.item()
            rec_loss = rec_loss.detach()
            
            cyc_loss = lambda_cyc*loss(netG( m_image(fake2),c_prev2),real2)
            cyc_loss.backward(retain_graph=True)
            loss_print['cyc_loss'] = cyc_loss.item()
            cyc_loss = cyc_loss.detach()
            
            Z_opt = m_image(real)
            rec_loss2 = lambda_idt*loss(netG2(Z_opt.detach(),z_prev),real)
            rec_loss2.backward(retain_graph=True)
            loss_print['rec_loss2'] = rec_loss2.item()
            rec_loss2 = rec_loss2.detach()

            cyc_loss2 = lambda_cyc*loss(netG2( m_image(fake),c_prev),real)
            cyc_loss2.backward(retain_graph=True)
            loss_print['cyc_loss2'] = cyc_loss2.item()
            cyc_loss2 = cyc_loss2.detach()

            optimizerG.step()

        if epoch % 25 == 0 or epoch == (opt.niter-1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            save_image(denorm(fake.data.cpu()), '%s/fake_sample.png' % (opt.outf))
            save_image(denorm(netG2(m_image(fake), z_prev).data.cpu()), '%s/cyc_sample.png' % (opt.outf))
            save_image(denorm(netG2(Z_opt.detach(), z_prev).data.cpu()), '%s/rec_sample.png' % (opt.outf))
            save_image(denorm(fake2.data.cpu()), '%s/fake_sample2.png' % (opt.outf))
            save_image(denorm(netG(m_image(fake2), z_prev2).data.cpu()), '%s/cyc_sample2.png' % (opt.outf))
            save_image(denorm(netG(Z_opt2.detach(), z_prev2).data.cpu()), '%s/rec_sample2.png' % (opt.outf))
            save_image(denorm(z_opt.data.cpu()), '%s/z_opt.png' % (opt.outf))
            save_image(denorm(noise.data.cpu()), '%s/noise.png' % (opt.outf))
            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))
            
            log = " Iteration [{}/{}]".format(epoch, opt.niter)
            for tag, value in loss_print.items():
                log += ", {}: {:.4f}".format(tag, value)
            print(log)

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG,netD,z_opt, netG2,netD2,z_opt2, opt)
    return z_opt,in_s,netG, z_opt2,in_s2,netG2       


def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = m_image(real_curr)
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z.detach(),1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
    return G_z


def cycle_rec(Gs,Gs2,Zs,reals,NoiseAmp,in_s,m_noise,m_image,opt, epoch):
    x_ab = in_s
    x_aba = in_s
    if len(Gs) > 0:
        count = 0
        for G,G2,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Gs2,Zs,reals,reals[1:],NoiseAmp):
            z = functions.generate_noise([3, Z_opt.shape[2] , Z_opt.shape[3] ], device=opt.device)
            z = z.expand(opt.bsz, 3, z.shape[2], z.shape[3])
            z = m_noise(z)
            x_ab = x_ab[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
            x_ab = m_image(x_ab)
            z_in = noise_amp*z+m_image(real_curr)
            x_ab = G(z_in.detach(),x_ab)
            
            x_aba = G2(x_ab,x_aba)
            x_ab = imresize(x_ab.detach(),1/opt.scale_factor,opt)
            x_ab = x_ab[:,:,0:real_next.shape[2],0:real_next.shape[3]]
            x_aba = imresize(x_aba.detach(),1/opt.scale_factor,opt)
            x_aba = x_aba[:,:,0:real_next.shape[2],0:real_next.shape[3]]
            count += 1
    return x_ab, x_aba


def init_models(opt):
    # generator initialization
    netG = models.GeneratorConcatSkip2CleanAddAlpha(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    # discriminator initialization
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    # generator2 initialization
    netG2 = models.GeneratorConcatSkip2CleanAddAlpha(opt).to(opt.device)
    netG2.apply(models.weights_init)
    if opt.netG2 != '':
        netG2.load_state_dict(torch.load(opt.netG2))
    print(netG2)

    # discriminator2 initialization
    netD2 = models.WDiscriminator(opt).to(opt.device)
    netD2.apply(models.weights_init)
    if opt.netD2 != '':
        netD2.load_state_dict(torch.load(opt.netD2))
    print(netD2)
    return netD, netG, netD2, netG2
