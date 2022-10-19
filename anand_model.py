#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:33:17 2021
@author: zen
"""

## Required import statements for program functionality.

import torch
import numpy as np
from scipy import io
import learner as ln
from learner.utils import mse, grad

class RBCData(ln.Data):
    
    ## Instantiates the dataset object.
    def __init__(self):
        super(RBCData, self).__init__()
        self.__init_data()
        
    ## Loads matlab files and normalizes x values.
    def generate(self):
        t = io.loadmat('data/t_1800.mat')['t']  # t step
        x = io.loadmat('data/Y_1801_34.mat')['Y'] # output matrix
        
        # Add noise
        #noise = np.random.normal(0, 0.05, [297, 15])   # Change to [318, 15] for 180s range
        #x = x + noise
        # Done adding noise
        
       # norm = np.max(x, axis = 0)[None,:] + 1e-8
        NDM = np.array([0.025, 10.0, 0.005, 10.0, 0.005, 90.0, 90.0, 10.0, 10.0, 170.0,
                 170.0, 10.0, 10.0, 1400.0, 1400.0, 10.0, 10.0, 10.0, 10.0,
                  0.7, 0.7, 0.7, 0.7, 0.7, 20.0, 20.0, 10.0, 10.0, 10.0, 7000.0,
                 7000.0, 2.5, 2.5, 3400.0])
        NDM = NDM[None,:]
        x /= NDM
        #x = np.apply_along_axis(lambda v: v/ np.array(NDM), axis=1, arr=x)
        print(x[0,:])         

        #norm[:,3] = norm[:,3]/10
        #x[:,3] = x[:,3] * 10
        # Add noise attempt 2 
        #noise = np.random.normal(0, 0.05, [297, 15]) # Change to [318, 15] for 180s range
        #x = x + noise
        # Done adding noise
        
        # Add noise attempt 3 - Random Seed
        #noise = np.random.normal(0, x.std(), [318,15]) * .05 # Change to [297, 15] for 180s range
        #x = x + noise
        # Done adding noise
        
        # Add noise - Set Seed
        #torch.manual_seed(0)
        #noisegen = torch.normal(0, x.std(), [297,15]) * .05 # Change to [318, 15] for 180s range; [297, 15] for 100s range
        #noise = noisegen.numpy()
        #x = x + noise
        # Done adding noise
        
        return t, x, NDM
    
    ## Instantiates the dataset object.
    def __init_data(self):
        self.X_train, self.y_train, self.norm = self.generate()
        self.X_test, self.y_test, _ = self.generate()
        
## This class takes in a neural net that optimizes based on minimizing the loss.
class RBCPINN(ln.nn.LossNN):
    
    ## Initializes the neural net.
    def __init__(self, net, norm, k_index, k_list, k_norm, x_index, lam1 = 1, lam2 = 1, lam3 = 1): 
        super(RBCPINN, self).__init__()
        self.net = net
        self.norm = norm
        self.k_index = k_index
        self.k_list = k_list
        self.k_norm = k_norm
        self.x_index = x_index
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.k_dim = 57 # total number of parameters
        self.K = self.__init_params()
        
    ## 
    def criterion(self, t, x):
        t = t.requires_grad_(True)
        x_pred = self.net(torch.cat([t,torch.exp(-t)],dim=-1))
        x_t = grad(x_pred, t).squeeze()
        #print(x_pred.device, x.device, self.x_index, x.shape, x_pred.shape)
        MSE2 = mse(x_pred[...,self.x_index], x[...,self.x_index])
        MSE3 = mse(x_pred[[0,-1]], x[[0,-1]])
        x_pred[...,self.x_index] = x[...,self.x_index]
       # MSE1 = mse(x_t[:,3], self.f(x_pred)[:,3])
        MSE1 = mse(x_t, self.f(x_pred))
        #return self.lam2 * MSE2 + self.lam1 * (MSE1 + MSE3)
        return self.lam2 * MSE2 + self.lam1 * MSE1 + self.lam3 * MSE3
    
    ## Defines the governing ODES
    def f(self, x):
        dx = []
        K = [torch.exp(k) for k in self.K]
        K = [K[i] * self.k_norm[i] for i in range(self.k_dim)]
        
        norm = torch.tensor(self.norm, dtype = self.dtype, device = self.device)
        x = x * norm
        #print(torch.sum(x))
        TIM = 1800
        NDM = [0.025, 10.0, 0.005, 10.0, 0.005, 90.0, 90.0, 10.0, 10.0, 170.0,
       170.0, 10.0, 10.0, 1400.0, 1400.0, 10.0, 10.0, 10.0, 10.0,
       0.7, 0.7, 0.7, 0.7, 0.7, 20.0, 20.0, 10.0, 10.0, 10.0, 7000.0,
       7000.0, 2.5, 2.5, 3400.0]

        # coagulation 34 ODE equations
        dx.append(TIM*(-K[0]*x[...,0]*x[...,1] + K[1]*x[...,2] - K[2]*x[...,0]*x[...,3] + K[3]*x[...,4])) # TF
        dx.append((TIM*(-K[0]*x[...,0]*x[...,1] + K[1]*x[...,2] - K[4]*x[...,4]*x[...,1] - K[5]*x[...,10]*x[...,1] - K[6]*x[...,14]*x[...,1])/NDM[1])) # VII
        dx.append((TIM*(K[0]*x[...,0]*x[...,1] - K[1]*x[...,3]))/NDM[2]) # TF:VII
        dx.append((TIM*(-K[2]*x[...,0]*x[...,3] + K[3]*x[...,4] + K[4]*x[...,4]*x[...,1] + K[5]*x[...,10]*x[...,1] + K[6]*x[...,6]*x[...,14]))/NDM[3]) #VIIa
        dx.append((TIM*(K[2]*x[...,0]*x[...,3] - K[3]*x[...,4] - K[20]*x[...,32]*x[...,4] - K[24]*x[...,33]*x[...,4]))/NDM[4]) # TF:VIIa
        dx.append((TIM*(((-K[9]*x[...,4]*x[...,5])/(K[10]+x[...,5])) - K[49]*250*x[...,18]*x[...,5] + K[50]*x[...,7]))/NDM[5]) # IX
        dx.append(TIM*(((K[9]*x[...,4]*x[...,5])/(K[10]+x[...,5])) - (K[49]*(550*x[...,18]*x[...,6] + K[50]*x[...,8] - K[22]*x[...,33]*x[...,6])/NDM[6]))) # IXa        
        dx.append((TIM*(K[49]*250*x[...,18]*x[...,5] - K[50]*x[...,7]))/NDM[7]) # IX(m)
        dx.append((TIM*(-K[24]*x[...,22]*x[...,8] - K[26]*x[...,23] + K[49]*550*x[...,18]*x[...,6] - K[50]*x[...,8]))/NDM[8]) # IXa(m)
        dx.append((TIM*((-K[7]*x[...,4]*x[...,9])/(K[8] + x[...,9]) + K[46]*x[...,11] - K[45]*2700*x[...,18]*x[...,9]))/NDM[9]) # X
        dx.append((TIM*(K[7]*x[...,4]*x[...,9])/(K[8] + x[...,9]) - K[18]*x[...,31]*x[...,10] + K[19]*x[...,32] - K[21]*x[...,33]*x[...,10] + K[46]*x[...,12] - K[45]*2700*x[...,18]*x[...,10])/NDM[10]) # Xa
        dx.append((TIM*((-K[27]*x[...,24]*x[...,12])/(K[28] + x[...,11]) - K[46]*x[...,11] - K[45]*2700*x[...,18]*x[...,10]))/NDM[11]) # X(m)
        dx.append((TIM*((-K[27]*x[...,24]*x[...,12])/(K[28] + x[...,11]) - K[29]*x[...,12]*x[...,27] + K[30]*x[...,28] - K[46]*x[...,12] + K[45]*2700*x[...,18]*x[...,13]))/NDM[12]) # Xa(m)
        dx.append((TIM*(-K[11]*x[...,10]*x[...,13] - K[48]*x[...,15] - K[47]*2000*x[...,18]*x[...,13]))/NDM[13]) # II
        dx.append((TIM*(K[11]*x[...,10]*x[...,13] - K[23]*x[...,33]*x[...,14] + K[48]*x[...,16] - K[47]*2000*x[...,18]*x[...,14]))/NDM[14]) # IIa        
        dx.append((TIM*((-K[31]*x[...,28]*x[...,15])/(K[32] + x[...,15]) - K[48]*x[...,15] - K[47]*2000*x[...,18]*x[...,13]))/NDM[15]) # II(m)
        dx.append(((TIM*(K[31]*x[...,28]*x[...,15])/(K[32] + x[...,15]) - K[48]*x[...,16] - K[47]*2000*x[...,18]*x[...,14]))/NDM[16]) # IIa(m)        
        dx.append(TIM*(((-K[55]*x[...,17]*x[...,18] - K[56]*x[...,17]*x[...,14]))/(1 + x[...,14]))/NDM[17]) # PL
        dx.append(TIM*(((-K[55]*x[...,17]*x[...,18] + K[56]*x[...,17]*x[...,14]))/(1 + x[...,14]))/NDM[18]) # AP
        dx.append(TIM*((-K[12]*x[...,15]*x[...,19])/(K[13] + x[...,19]) - K[51]*750*x[...,18]*x[...,19] + K[52]*x[...,21])/NDM[19]) # VIII
        dx.append(TIM*((K[12]*x[...,15]*x[...,19])/(K[13] + x[...,19]) - K[51]*750*x[...,18]*x[...,20] + K[52]*x[...,22] - K[35]*x[...,20])/NDM[20]) # VIIIa        
        dx.append(TIM*(((K[39]*x[...,16]*x[...,21])/(K[40] + x[...,21])) - ((K[33]*750*x[...,12]*x[...,21])/(K[34] + x[...,21])) + K[51]*750*x[...,18]*x[...,19] - K[52]*x[...,21])/NDM[21]) # VIII(m)        
        dx.append(TIM*(((K[39]*x[...,16]*x[...,21])/(K[40] + x[...,21])) - ((K[33]*750*x[...,12]*x[...,21])/(K[34] + x[...,21])) + K[51]*750*x[...,18]*x[...,20] - K[52]*x[...,21] - K[25]*x[...,22]*x[...,8] + K[26]*x[...,23])/NDM[22]) # VIIIa(m)
        dx.append(TIM*((K[25]*x[...,22]*x[...,8] - K[26]*x[...,23]))/NDM[23]) # IXa:VIIIa
        dx.append(TIM*(((-K[14]*x[...,14]*x[...,24])/(K[15] + x[...,24]) - K[53]*2700*x[...,18]*x[...,24] + K[54]*x[...,26]))/NDM[24]) # V
        dx.append(TIM*(((-K[14]*x[...,14]*x[...,24])/(K[15] + x[...,24]) - K[53]*2700*x[...,18]*x[...,25] + K[54]*x[...,27] - K[38]*x[...,25]))/NDM[25]) # Va
        dx.append(TIM*(((-K[41]*x[...,16]*x[...,26])/(K[42] + x[...,26]) - (K[36]*x[...,12]*x[...,26])/(K[37] + x[...,26]) + K[53]*2700*x[...,18]*x[...,24] - K[54]*x[...,26]))/NDM[26]) # V(m)        
        dx.append(TIM*(((-K[41]*x[...,16]*x[...,26])/(K[42] + x[...,26]) - (K[36]*x[...,12]*x[...,26])/(K[37] + x[...,26]) - K[29]*x[...,12]*x[...,27] + K[30]*x[...,28] + K[53]*2700*x[...,18]*x[...,25] - K[54]*x[...,27]))/NDM[27]) # Va(m)        
        dx.append(TIM*((K[29]*x[...,12]*x[...,27] - K[30]*x[...,28]))/NDM[28]) # Xa:Va
        dx.append(TIM*((-K[16]*x[...,14]*x[...,29])/(K[17] + x[...,29]))/NDM[29]) # I
        dx.append(TIM*((K[16]*x[...,14]*x[...,29])/(K[17] + x[...,29]))/NDM[30]) # Ia
        dx.append(TIM*(-K[18]*x[...,10]*x[...,31] + K[19]*x[...,32])/NDM[31]) # TFPI
        dx.append(TIM*(K[18]*x[...,10]*x[...,31] - K[19]*x[...,32] - K[20]*x[...,4]*x[...,32])/NDM[32]) # xa:TFPI
        dx.append(TIM*(-x[...,33]*(K[21]*x[...,10] + K[22]*x[...,6] + K[23]*x[...,14] + K[24]*x[...,4]) )/NDM[33]) # ATIII

        dx = torch.stack(dx, dim = -1)/norm
        #print(torch.sum(dx), K)
        return dx
    ## 
    def predict(self, t, returnnp=False):
        x = self.net(torch.cat([t,torch.exp(-t)],dim=-1))
        if returnnp:
            x = x.detach().cpu().numpy()
        return x
    
    ## Returns parameters
    def __init_params(self):
        params = torch.nn.ParameterList()
        for i in range(self.k_dim):
            if i not in self.k_index:
                ki = torch.tensor(self.k_list[i], dtype = self.dtype, device = self.device)
                params.append(torch.nn.Parameter(torch.log(ki), requires_grad = False))
            else:
                ki = torch.randn(1, dtype = self.dtype, device = self.device)[0]
                params.append(torch.nn.Parameter(ki, requires_grad = True))
        return params  
    
def callback(data, net):
    t, x = data.get_batch(None)
    t = t.requires_grad_(True)
    x_pred = net.net(torch.cat([t,torch.exp(-t)],dim=-1))
    x_t = grad(x_pred, t).squeeze()
    MSE2 = mse(x_pred[...,net.x_index], x[...,net.x_index])
    MSE3 = mse(x_pred[[0,-1]], x[[0,-1]])
    x_pred[...,net.x_index] = x[...,net.x_index]
    MSE1 = mse(x_t, net.f(x_pred))
    MSE_list = [MSE1.item(),MSE2.item(),MSE3.item()]
    for i in net.x_index:
        MSE = mse(x_pred[...,i], x[...,i])
        MSE_list.append(MSE.item())
    for i in range(x_pred.shape[-1]):
        MSE = mse(x_t[...,i], net.f(x_pred)[...,i])
        MSE_list.append(MSE.item())
    return MSE_list

def plot(data, net):
    import matplotlib.pyplot as plt
    t, y = data.get_batch_test(None)
    x = net.predict(t, True)
    y = y.detach().cpu().numpy()
    x = x * data.norm
    y = y * data.norm
    
    # Output factor - ODEs
    # Need double check 
    #name_list = ['TF','VII', 'TF:VII', 'VIIa', 'TF:VIIa', 'IX', 'IXa', 'IXm', 'IXam',
    #'X', 'Xa', 'Xm', 'Xam', 'II', 'IIa', 'IIm', 'IIam', 'PL', 'AP', 'VIII', 'VIIIa', 'VIIIm',
    #'VIIIam', 'IXa:VIIIa', 'V', 'Va', 'Vm', 'Vam', 'Xa:Va', 'I', 'Ia', 'TFPI', 'Xa:TFPI', 'ATIII'] # output pdf format
   

    name_list = ['TF', 'TF:VII', 'TF:VIIa', 'X', 'Xa', 'Xm', 'Xam', 'II', 'IIa', 'IIm', 'IIam', 'PL', 'AP','I', 'Ia'] # output pdf format
    plt.figure(figsize=[4.8 * 5, 4.8 * 3])
    for i in range(len(name_list)):
        plt.subplot(3,5,i+1)
        plt.plot(t.detach().cpu().numpy(), x[...,i], 'r', label = 'PINN')
        plt.plot(t.detach().cpu().numpy(), y[...,i], 'b', label = 'Ground Truth')
        if i in net.x_index:
            plt.plot(t.detach().cpu().numpy(), y[...,i], 'b.', label = 'Data')
        plt.title('{} (X{})'.format(name_list[i], i+1), fontsize=18)
        plt.legend()
    plt.tight_layout()
    plt.savefig('pinn.pdf')
    k_pred = np.array([np.exp(k.item()) for k in net.K])[net.k_index]
    k_true = np.array(net.k_list)[net.k_index]
    np.savetxt('k_learned', k_pred)
    np.savetxt('k_true', k_true)
    
def plot2(path):
    import matplotlib.pyplot as plt
    loss = np.loadtxt('outputs/'+path+'/loss.txt')
    if len(loss.shape) == 1:
        loss = loss.reshape([1,-1])
    iters = loss[:,0]
    plt.figure(figsize = [6.4, 4.8*3])
    plt.subplot(311)
    plt.plot(iters, loss[:,3], label = r'$L_{ode}$')
    plt.legend()
    plt.yscale('log',base=10)
    plt.subplot(312)
    plt.plot(iters, loss[:,4], label = r'$L_{data}$')
    plt.legend()
    plt.yscale('log',base=10)
    plt.subplot(313)
    plt.plot(iters, loss[:,5], label = r'$L_{aux}$')
    plt.yscale('log',base=10)
    plt.legend()
    # The rest columns of loss are the individual losses, you can plot them below
    plt.savefig('loss.pdf')
    plt.tight_layout()

def main():
    device = 'gpu' # 'cpu' or 'gpu' 'cuda'
    # fnn
    depth = 5
    width = 60
    activation = 'tanh'  # Standard is tanh
    # training
    lr = 0.001  # 0.001
    iterations =2000   # Set to 0 to return to previous model
    lbfgs_steps = 1000
    print_every = 1000
    batch_size = None
    
    # All k parameters for Anand model
    k1   = 3.2;              #nM^{-1}s^{-1}    %% binding of TF & VII 0.0458;%
    k2   = 3.1;              #s^{-1}           %% dissociation of TF:VII 0.0183;%
    k3   = 0.023;            #nM^{-1}s^{-1}    %% binding of TF & VIIa 0.0458;%
    k4   = 3.1;              #s^{-1}           %% dissociation of TF:VIIa 0.0183;%
    k5   = 4.4;              #nM^{-1}s^{-1}    %% auto-activation of VII (H&M,2002)
    k6   = 0.013;            #nM^{-1}s^{-1}    %% Xa-activation of VII
    k7   = 2.3;              #nM^{-1}s^{-1}    %% IIa-activation of VII
    k8   = 69.0/60.0;        #s^{-1}           %% TF:VIIa activation of X (103.0/60.0; AM,2008) (Mann et al 1990,Blood J)
    K8M  = 450.0;            #nM               %% TF:VIIa activation of X (240.0; AM,2008) (Mann et al 1990)
    k9   = 15.6/60.0;        #s^{-1}           %% TF:VIIa activation of IX (32.4/60.0; AM,2008) (Mann et al 1990,Blood J) 
    K9M  = 243.0;            #nM               %% TF:VIIa activation of IX (24.0; AM,2008) (Mann et al 1990) 
    k10  = 7.5;              #nM^{-1}s^{-1}    %% Xa-activation of II
    k11  = 54.0/60.0;        #s^{-1}           %% IIa-activation of VIII(194.4/60.0; AM,2008) (Hill-Eubanks & Lollar 1990)(modified 6/20/2016)
    K11M = 147.0;            #nM               %112000;
    k12  = 0.233;            #s^{-1}           %% IIa-activation of V (%27.0/60.0; AM,2008) (Monkovic & Tracy, 1990)(modified 6/20/2016)
    K12M = 71.7;             #nM               %140.5;
    kf   = 59.0;             # s^{-1}          %% IIa-activation of fibrinogen (AM,2008)
    KfM  = 3160.0;           # nM
    k13  = 4.381;            #nM^{-1}s^{-1}   %% binding of Xa with TFPI (modified 6/20/2016)
    k14  = 5.293;            #s^{-1}          %% dissociation of Xa:TFPI
    k15  = 0.05;             #nM^{-1}s^{-1}   %% Xa:TFPI inactivation of TF:VIIa
    k16  = 1.83/60.0;        #%(fminc)  %nM^{-1}s^{-1}   %% ATIII inactivation of Xa % (1.83e-04/60.0,Wb,2003); (1.5e-06, HM 2002); AM,2008 0.347/60.0 (Panteleev 2006)
    k17  = 1.34/60.0;        #nM^{-1}s^{-1}   %% ATIII inactivation of IXa % (1.34e-05/60.0,Wb,2003); (4.9e-07, HM 2002); AM,2008 0.0162/60.0 (Panteleev 2006)
    k18  = 1.79;             #%(fminc)   %nM^{-1}s^{-1}   %% ATIII inactivation of IIa % (2.89e-04/60.0,Wb,2003); (7.1e-06, HM 2002); AM,2008 0.0119 11.56e-03 (Panteleev 2006)(modified 6/20/2016)
    k19  = 4.5;              #nM^{-1}s^{-1}   %% ATIII inactivation of TF:VIIa (HM 2002 2.3e-07)(Lawson et al.,1993,4.5e-07 no HS, 5.6e-06 with HS)
    k20  = 0.01;             #nM^{-1}s^{-1}   %% binding of IXa^{m} and VIIIa^{m} (modified 6/20/2016)
    k21  = 5.0;              #s^{-1}          %% dissociation of IXa{m}:VIIIa{m} 0.01;%
    k22  = 500.0/60.0;       #s^{-1}          %% IXa:VIIIa activation of X^{m}(2391.0/60.0 AM 2008) (Mann et al 1990) (20, KnF)
    k23  = 63.0;             #nM              %% IXa:VIIIa activation of X^{m}( %160.0 AM 2008)  (Mann et al 1990) (160, KnF)
    k24  = 0.4;              #nM^{-1}s^{-1}   %% binding of Xa^{m} and Va^{m} 0.1;%
    k25  = 0.2;              #s^{-1}          %% dissociation of Xa:Va  0.01;%
    k26  = 1344.0/60.0;      #s^{-1}          %% Xa:Va activation of II^{m}(AM 2008)  (1800.0/60.0; % (Krishnaswamy et al 1990)) Revert to original (16/9/15) (30, KnF)
    k27  = 1060.0;           #nM              %% Xa:Va activation of II^{m}(AM 2008) (1000.0; % (Krishnaswamy et al 1990)) (300, KnF)
    k28  = 0.023;            #s^{-1}          %% Xa^{m}-activation of VIII^{m}
    K28M = 20.0;             #nM              %% Xa^{m}-activation of VIII^{m}(KnF,2001)
    h8   = 0.0037;           #s^{-1}          %% spontaneous decay of VIIIa
    k29  = 0.046;            #s^{-1}          %% Xa^{m}-activation of V^{m}
    K29M = 10.4;             #nM              %% Xa^{m}-activation of V^{m}(KnF,2001)
    h5   = 0.0028;           #s^{-1}          %% spontaneous decay of Va
    k30  = 0.9;              #s^{-1}          %% IIa^{m} activation of VIII^{m}(AM,2008)(modified 6/20/2016)
    K30M = 147.0;            #nM
    k31  = 0.233;            #s^{-1}          %% IIa^{m} activation of V^{m}(AM,2008)(modified 6/20/2016)
    K31M = 71.7;             #nM
    b11 = 0.01;              # nM^{-1}s^{-1}   %% platelet-binding of fXI/XIa; FOGELSON et al., 2011
    d11 = 0.1;               # sec^{-1}        %% dissociation of fXI/XIa from platelets
    b10 = 0.029;             # nM^{-1}s^{-1}   %% platelet-binding of fX/Xa (Krishnaswamy et al. 1998)
    d10 = 3.3;               # sec^{-1}        %% dissociation of fX/Xa from platelets
    b2  = 0.01;              # nM^{-1}s^{-1}   %% platelet-binding of fII/IIa
    d2  = 5.9;               # sec^{-1}        %% dissociation of fII/IIa from platelets
    b9  = 0.01;              # nM^{-1}s^{-1}   %% platelet-binding of fIX/IXa
    d9  = 0.0257;            # sec^{-1}        %% dissociation of fIX/IXa from platelets
    b8  = 4.3;               # nM^{-1}s^{-1}   %% platelet-binding of fVIII/VIIIa (Raut et al 1999)
    d8  = 2.46;              # sec^{-1}        %% dissociation of fVIII/VIIIa from platelets (Raut et al 1999)
    b5  = 0.057;             # nM^{-1}s^{-1}   %% platelet-binding of fV/Va (Krishnaswamy et al. 1998)
    d5  = 0.17;              # sec^{-1}        %% dissociation of fV/Va from platelets
    kpp = 0.3;               # nM^{-1}s^{-1}   %% platelet-activation of platelet (KnF 2001)
    kp2 = 0.37;              # s^{-1}          %% thrombin-activation of platelet 5.4/60.0;%

    # TIM 
    TIM = 1800
    NDM = [0.025, 10.0, 0.005, 10.0, 0.005, 90.0, 90.0, 10.0, 10.0, 170.0,
       170.0, 10.0, 10.0, 1400.0, 1400.0, 10.0, 10.0, 10.0, 10.0,
       0.7, 0.7, 0.7, 0.7, 0.7, 20.0, 20.0, 10.0, 10.0, 10.0, 7000.0,
       7000.0, 2.5, 2.5, 3400.0]

    # initial condition for the integration
    # 'TF','VII', 'TF:VII', 'VIIa', 'TF:VIIa', 'IX', 'IXa', 'IXm', 'IXam',
    # 'X', 'Xa', 'Xm', 'Xam', 'II', 'IIa', 'IIm', 'IIam', 'PL', 'AP', 'VIII', 'VIIIa', 'VIIIm',
    # 'VIIIam', 'IXa:VIIIa', 'V', 'Va', 'Vm', 'Vam', 'Xa:Va', 'I', 'Ia', 'TFPI', 'Xa:TFPI', 'ATIII'
    x_initial = [0.005, 10.0, 0.0, 0.001, 0.0, 90.0, 0.009, 0.0, 0.0, 170.0, 0.017,
      0.0, 0.0, 1400.0, 0.14, 0.0, 0.0, 10.0, 0.0010, 0.7, 0.00007,
      0.0, 0.0, 0.0, 20.0, 0.002, 0.0, 0.0, 0.0, 7000.0, 0.70, 2.5, 0.0,
      3400.0]
    
    k_index = [16,18,20] ## ASSUMING SOME UNKNOWN
    
    #k_index = [7, 8, 10, 13, 14, 15, 16, 21, 22] ## ASSUMING SOME UNKNOWN W/O KNOWN EXP PARAMETER
    #k_index = [9]    
    # K  ----- rate constant change here   

    k_list = [k1, k2, k3, k4, k5, k6, k7, k8, K8M, k9, K9M, k10, k11, K11M, k12,
    K12M, kf, KfM, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25,
    k26, k27, k28, K28M, h8, k29, K29M, h5, k30, K30M, k31, K31M, b11, d11, b10, d10, 
    b2, d2, b9, d9, b8, d8, b5, d5, kpp, kp2]

    #k_norm = [1, 1, 1, 1, 1,
    #          1, 1, 1, 1, 1e-6,
    #          1, 1e-6, 1e-8, 1, 1,
    #          1, 1, 1, 1, 1,

    k_norm = [1e-3, 1e-3, 1, 1e-3, 1e-4, 1, 1e-5, 1, 1, 1, 1, 1e-6, 1, 1, 1,
    1, 1, 1, 1, 1e-8, 1, 1e-4, 1e-5, 1e-4, 1e-7, 1, 1e-3, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1e-3, 1e-3, 1, 1, 1, 1]

    # 34 coagulation factors - output them all
    #x_index = [0,1,2,3,4,5,6,7,8]
    x_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]

    data = RBCData()
    fnn = ln.nn.FNN(2, len(x_initial), depth, width, activation)
    net = RBCPINN(fnn, data.norm, k_index, k_list, k_norm, x_index, lam1 = 0.1, lam2 = 1)
    #net = torch.load('outputs/test1/model_best.pkl', map_location=torch.device('cpu'))
    #net.lam1 = 0.01
    
    path = 'test1'
    args = {
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'lbfgs_steps': lbfgs_steps,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': True,
        'callback': callback,
        'path': path,
        'dtype': 'float',
        'device': device,
    }
    
    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    
    plot(data, ln.Brain.Best_model())
    #plot2(path)
    
if __name__ == '__main__':
    main()
