#!/usr/bin/env python3
###########################################################
# Authors: Joel Anyanti, Jui-Chieh Chang, Alex Condotti
# Carnegie Mellon Univerity
# 11-785 (Introduction to Deep Learning)
#
# model.py
###########################################################
# Imports
###########################################################
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time

from config import *
###########################################################
# Model Helpers
###########################################################

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.shape
    y_shapes = y.shape
    y2 = y.expand(x_shapes[0],y_shapes[1],x_shapes[2],x_shapes[3])

    return torch.cat((x, y2),1)

def conv_prev_concat(x, y):
        """Concatenate conditioning vector on feature map axis."""
        x_shapes = x.shape
        y_shapes = y.shape
        if x_shapes[2:] == y_shapes[2:]:
            y2 = y.expand(x_shapes[0],y_shapes[1],x_shapes[2],x_shapes[3])

            return torch.cat((x, y2),1)

        else:
            print(x_shapes[2:])
            print(y_shapes[2:])


###########################################################
# Model Subunits
###########################################################

class LConv2d(nn.Module):
    def __init__(self, c_in, c_out, k, s, p):
        super(LConv2d, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x

class ConvTranspose2d(nn.Module):
    def __init__(self, c_in, c_out, k, s, p):
        super(ConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

###########################################################
# Model
###########################################################
class Generator(nn.Module):
    def __init__(self, gf_dim=64, nz=100, pitch_range=PITCH, bar_length=BAR,
                 is_chord=False, chord_dims=None):
        super(Generator, self).__init__()

        # Define class properties
        self.gf_dim = gf_dim
        self.nz = nz # length of input vector 'z' (noise signal)
        self.pitch_range = pitch_range
        self.bar_length = bar_length
        self.is_chord = is_chord and chord_dims != None
        self.num_filters = 256
        self.w_final = self.bar_length // 8
        self.num_linear = self.gf_dim*self.w_final*2*1
        self.linear_in = [self.nz, 1024]
        self.linear_out = (1024, self.num_linear)
        self.transpose_in = 384

        # Add properties for chord conditioning
        if self.is_chord:
          self.c_feat_dim = chord_dims[0]
          self.c_time_dim = chord_dims[1]

          self.c_flat_dim = self.c_time_dim * self.c_feat_dim
          self.linear_in[0] += self.c_flat_dim
          self.linear_in[1] += self.c_flat_dim
          self.transpose_in += self.c_flat_dim


        # Noise Projection Layer
        self.h0_prev = LConv2d(c_in=1, c_out=self.num_filters, k=(1,pitch_range), s=(1,2), p=0)
        self.h1_prev = LConv2d(c_in=self.num_filters, c_out=self.num_filters, k=(2,1), s=(2,2), p=0)
        self.h2_prev = LConv2d(c_in=self.num_filters, c_out=self.num_filters, k=(2,1), s=(2,2), p=0)
        self.h3_prev = LConv2d(c_in=self.num_filters, c_out=self.num_filters, k=(2,1), s=(2,2), p=0)

        # Conditions Layer
        self.h1 = ConvTranspose2d(c_in=self.transpose_in, c_out=pitch_range, k=(2,1), s=(2,2), p=0)
        self.h2 = ConvTranspose2d(c_in=self.transpose_in, c_out=pitch_range, k=(2,1), s=(2,2), p=0)
        self.h3 = ConvTranspose2d(c_in=self.transpose_in, c_out=pitch_range, k=(2,1), s=(2,2), p=0)
        self.h4 = ConvTranspose2d(c_in=self.transpose_in, c_out=1, k=(1,pitch_range), s=(1,2), p=0)

        # Linear Transformation layer
        self.linear1 = Linear(self.linear_in[0], self.linear_out[0])
        self.linear2 = Linear(self.linear_in[1], self.linear_out[1])

        self.sigmoid = nn.Sigmoid()

    def forward(self, z, prev_x, y=None): #N x C x H x W

        prev_x = prev_x.permute(0,1,3,2) #N x C x W x H
        b_size = prev_x.shape[0] #N

        h0_prev = self.h0_prev(prev_x)  #N x C0 x W x 1
        h1_prev = self.h1_prev(h0_prev) #N x C1 x W/2 x 1
        h2_prev = self.h2_prev(h1_prev) #N x C2 x W/4 x 1
        h3_prev = self.h3_prev(h2_prev) #N x C3 x W/8 x 1

        if self.is_chord:
          yb = y.view(b_size, self.c_flat_dim, 1, 1) #N x Cf x 1 x 1
          yf = y.view(b_size, self.c_flat_dim) #N x Cf
          z = torch.cat((z,yf), 1) #N x (nz + Cf)


        h0 = self.linear1(z) #N x L1
        if self.is_chord: h0 = torch.cat((h0,yf), 1) #N x (ln1 + Cf)

        h1 = self.linear2(h0) #N x L2
        h1 = h1.view(b_size, self.gf_dim * 2, self.w_final, 1) #N x H x W/8 x 1
        if self.is_chord: h1 = conv_cond_concat(h1, yb) #N x (H + Cf) x W/8 x 1
        h1 = conv_prev_concat(h1, h3_prev) #N x (H + NF) x W/8 x 1

        h2 = self.h1(h1) #N x H x W/4 x 1
        if self.is_chord: h2 = conv_cond_concat(h2, yb) #N x (H + Cf) x W/4 x 1
        h2 = conv_prev_concat(h2, h2_prev) #N x (H + NF) x W/4 x 1

        h3 = self.h2(h2) #N x H x W/2 x 1
        if self.is_chord: h3 = conv_cond_concat(h3, yb) #N x (H + Cf) x W/2 x 1
        h3 = conv_prev_concat(h3, h1_prev) #N x (H + NF) x W/2 x 1

        h4 = self.h3(h3) #N x H x W x 1
        if self.is_chord: h4 = conv_cond_concat(h4, yb) #N x (H + Cf) x W x 1
        h4 = conv_prev_concat(h4, h0_prev) #N x (H + NF) x W x 1

        a_x = self.h4(h4) #N x 1 x W x H
        g_x = self.sigmoid(a_x) #N x 1 x W x H

        return g_x



class Discriminator(nn.Module):
    def __init__(self, df_dim=64, dfc_dim=1024, pitch_range=PITCH, bar_length=BAR,
                 is_chord=False, chord_dims=None):
        super(Discriminator, self).__init__()

        # Define class properties
        self.df_dim = df_dim
        self.dfc_dim = dfc_dim
        self.num_filters = df_dim
        self.pitch_range = pitch_range
        self.bar_length = bar_length
        self.is_chord = is_chord and chord_dims != None
        self.conv_in = 1
        self.kernel_h = 4
        self.kernel_w = 89
        self.w_final = bar_length - (self.kernel_h-1) * 3 # final W dimension after all conv layers
        self.h_final = 40 # final H dimension after all conv layers
        self.linear_in = self.df_dim * self.h_final * self.w_final # (conv kernel output (H,W))

        # Add properties for chord conditioning
        if self.is_chord:
          self.c_feat_dim = chord_dims[0]
          self.c_time_dim = chord_dims[1]

          self.c_flat_dim = self.c_time_dim * self.c_feat_dim
          self.conv_in += self.c_flat_dim

          self.kernel_h = 2
          self.kernel_w = self.pitch_range
          self.linear_in = 1784


          self.h0 = LConv2d(c_in=self.conv_in, c_out=self.conv_in, k=(self.kernel_h,self.kernel_w), s=2, p=0)
          self.h1 = LConv2d(c_in=27, c_out=77, k=(4,1), s=2, p=0)
          self.linear1 = nn.Linear(self.linear_in, 1024)
          self.linear2 = nn.Linear(1037, 1)

        else:
          self.h0 = LConv2d(c_in=self.conv_in, c_out=self.num_filters, k=(self.kernel_h,self.kernel_w), s=1, p=0)
          self.h1 = LConv2d(c_in=self.num_filters, c_out=self.num_filters, k=(4,1), s=1, p=0)
          self.h2 = LConv2d(c_in=self.num_filters, c_out=self.num_filters, k=(4,1), s=1, p=0)

          self.linear = nn.Linear(self.linear_in, 1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x, y= None, transform=False): #N x C x W x H
        if transform:
            x = x.permute(0,1,3,2) #N x C x W x H
        b_size = x.shape[0]

        if self.is_chord:
          yb = y.view(b_size, self.c_flat_dim, 1, 1) #N x Cf x 1 x 1
          yf = y.view(b_size, self.c_flat_dim) #N x Cf
          x = conv_cond_concat(x, yb) #N x Cf+1 x W x H

          h0 = self.h0(x)  #N x C0 x W/2 x 1
          h0 = conv_cond_concat(h0, yb) #N x 27 x W/2 x 1

          h1 = self.h1(h0) #N x 77 x 23 x H
          h1 = h1.reshape(b_size, -1) #N x 1771
          h1 = torch.cat((h1, yf),1) #N x 1784

          h2 = self.linear1(h1) #N x 1024
          h2 = torch.cat((h2,yf),1) #N x 1037

          out = self.linear2(h2)
          out_sigmoid = self.sigmoid(out) #N x 1

        else:
          h0 = self.h0(x)  #N x C0 x W-(K-1) x H
          h1 = self.h1(h0) #N x C0 x W-2*(K-1) x H
          h2 = self.h2(h1) #N x C0 x W-3*(K-1) x H
          h2 = h2.reshape(b_size, self.linear_in) #N x (NF x HF x WF)
          out = self.linear(h2) #N x 1

          out_sigmoid = self.sigmoid(out) #N x 1

        return out_sigmoid, out

###########################################################
# Training Functions
###########################################################
def collect_gc():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                del obj
        except:
            pass

def train(model, train_loader, opt, criterion, nz, device, epochs=25, is_chord=False):
    # Expand params
    modelG, modelD = model
    optG, optD = opt

    # Training data export
    training_run = ""

    # Establish convention for real and fake labels during training
    real_label = 0.9
    fake_label = 0.

    # Loss accumulators
    average_lossD = 0
    average_lossG = 0
    average_D_x   = 0
    average_D_G_z = 0

    lossD_list =  []
    lossD_list_all = []
    lossG_list =  []
    lossG_list_all = []
    D_x_list = []
    D_G_z_list = []

    modelG.train()
    modelD.train()
    for epoch in range(epochs):
        sum_lossG = 0
        sum_lossD = 0
        sum_D_x = 0
        sum_D_G_z = 0
        start_time = time.time()
        for i, data in enumerate(train_loader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            X_prev = data[0]
            X = data[1]

            if is_chord:
                chords = data[2]

            # train with real samples
            modelD.zero_grad()
            X = X.to(device)
            X_prev = X_prev.to(device)

            if is_chord:
                chords = chords.to(device)

            # Format batch
            b_size = X.size(0)
            label = torch.full((b_size,1), real_label, dtype=torch.float, device=device) # Create real labels

            # Forward pass real batch through D
            X = X.permute(0,1,3,2) # Permutate tensor to produce correct shape

            if is_chord:
              out, out_logits = modelD(X, chords)
            else:
              out, out_logits = modelD(X)

            # Calculate loss on all-real batch
            d_loss_real = criterion(out, label)

            # Calculate gradients for D in backward pass
            d_loss_real.backward(retain_graph=True)
            D_x = out.mean().item()
            sum_D_x += D_x

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, device=device)

            # Generate fake image batch with G
            if is_chord:
              fake = modelG(noise, X_prev, chords)
              label.fill_(fake_label)
              # Classify all fake batch with D
              out, out_logits = modelD(fake.detach(), chords)
            else:
              fake = modelG(noise, X_prev)
              label.fill_(fake_label)
              # Classify all fake batch with D
              out, out_logits = modelD(fake.detach())

            # Calculate D's loss on the all-fake batch
            d_loss_fake = criterion(out, label)

            # Calculate the gradients for this batch
            d_loss_fake.backward(retain_graph=True)
            D_G_z1 = out.mean().item()

            # Add the gradients from the all-real and all-fake batches
            errD = d_loss_real + d_loss_fake
            errD = errD.item()

            # Update D
            sum_lossD += errD
            optD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            modelG.zero_grad()
            label.fill_(real_label) # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            if is_chord:
              out, out_logits = modelD(fake, chords)
            else:
              out, out_logits = modelD(fake)

            # Calculate G's loss based on this output
            errG = criterion(out, label)
            sum_lossG += errG.data

            # Calculate gradients for G
            errG.backward(retain_graph=True)

            D_G_z2 = out.mean().item()
            sum_D_G_z += D_G_z2
            # Update G
            optG.step()

            ############################
            # (3) Update G network again: maximize log(D(G(z)))
            # Done to mitigate strength of Discriminator model
            ###########################
            modelG.zero_grad()
            label.fill_(real_label)

            # Since we just updated D, perform another forward pass of all-fake batch through D
            if is_chord:
              fake = modelG(noise, X_prev, chords)
              out, out_logits = modelD(fake, chords)
            else:
              fake = modelG(noise, X_prev)
              out, out_logits = modelD(fake)

            # Calculate G's loss based on this output
            errG = criterion(out, label)

            # Calculate gradients for G
            errG.backward(retain_graph=True)

            D_G_z2 = out.mean().item()
            sum_D_G_z += D_G_z2
            # Update G
            optG.step()

            #if epoch % 5 == 0 and i % 200 == 0:
            if i % 200 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch+1, epochs, i, len(train_loader),
                         errD, errG, D_x, D_G_z1, D_G_z2))

            del X
            del X_prev
            del out
            del out_logits
            del label
            del fake
            del noise
            del errD
            del errG
            del d_loss_real
            del d_loss_fake
            del D_x
            del D_G_z1
            del D_G_z2

            if is_chord:
                del chords

            collect_gc()

            torch.cuda.empty_cache()

        average_lossD = (sum_lossD / len(train_loader))
        average_lossG = (sum_lossG / len(train_loader))
        average_D_x = (sum_D_x / len(train_loader))
        average_D_G_z = (sum_D_G_z / len(train_loader))

        lossD_list.append(average_lossD)
        lossG_list.append(average_lossG)
        D_x_list.append(average_D_x)
        D_G_z_list.append(average_D_G_z)

        print("took {}s".format(time.time()-start_time))
        banner_str = '==> Epoch: {} average_lossD: {:.10f}, average_lossG: {:.10f}, average D(x): {:.10f}, average D(G(z)): {:.10f} \n'.format(epoch+1, average_lossD,average_lossG,average_D_x, average_D_G_z)
        training_run += banner_str
        print(banner_str)


    return training_run
