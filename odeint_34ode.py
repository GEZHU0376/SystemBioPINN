# ODEint solving system of ODEs for the 34 equations coagulation model
# Date: 1-27-2022
# Name: Ge Zhu

import numpy as np
import torch
import matplotlib as plt
from scipy import io
from scipy.integrate import odeint

# time span
TIM = 1800

# normalization?        
NDM = [0.025, 10.0, 0.005, 10.0, 0.005, 90.0, 90.0, 10.0, 10.0, 170.0,
       170.0, 10.0, 10.0, 1400.0, 1400.0, 10.0, 10.0, 10.0, 10.0,
       0.7, 0.7, 0.7, 0.7, 0.7, 20.0, 20.0, 10.0, 10.0, 10.0, 7000.0,
       7000.0, 2.5, 2.5, 3400.0]

# initial conditions
x_initial = [0.005, 10.0, 0.0, 0.001, 0.0, 90.0, 0.009, 0.0, 0.0, 170.0, 0.017,
      0.0, 0.0, 1400.0, 0.14, 0.0, 0.0, 10.0, 0.0010, 0.7, 0.00007,
      0.0, 0.0, 0.0, 20.0, 0.002, 0.0, 0.0, 0.0, 7000.0, 0.70, 2.5, 0.0,
      3400.0]

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

# k list used for indexing 
K = [k1, k2, k3, k4, k5, k6, k7, k8, K8M, k9, K9M, k10, k11, K11M, k12,
    K12M, kf, KfM, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25,
    k26, k27, k28, K28M, h8, k29, K29M, h5, k30, K30M, k31, K31M, b11, d11, b10, d10, 
    b2, d2, b9, d9, b8, d8, b5, d5, kpp, kp2]

# ODEs 
# coagulation 34 ODE equations
d1dt = (TIM*(-K[0]*x[...,0]*x[...,1] + K[1]*x[...,2] - K[2]*x[...,0]*x[...,3] + K[3]*x[...,4])) # TF   
d2dt = ((TIM*(-K[0]*x[...,0]*x[...,1] + K[1]*x[...,2] - K[4]*x[...,4]*x[...,1] - K[5]*x[...,10]*x[...,1] - K[6]*x[...,14]*x[...,1])/NDM[1])) # VII
d3dt = ((TIM*(K[0]*x[...,0]*x[...,1] - K[1]*x[...,3]))/NDM[2]) # TF:VII
d4dt = ((TIM*(-K[2]*x[...,0]*x[...,3] + K[3]*x[...,4] + K[4]*x[...,4]*x[...,1] + K[5]*x[...,10]*x[...,1] + K[6]*x[...,6]*x[...,14]))/NDM[3]) #VIIa
d5dt = ((TIM*(K[2]*x[...,0]*x[...,3] - K[3]*x[...,4] - K[20]*x[...,32]*x[...,4] - K[24]*x[...,33]*x[...,4]))/NDM[4]) # TF:VIIa
d6dt = ((TIM*(((-K[9]*x[...,4]*x[...,5])/(K[10]+x[...,5])) - K[49]*250*x[...,18]*x[...,5] + K[50]*x[...,7]))/NDM[5]) # IX
d7dt = (TIM*(((K[9]*x[...,4]*x[...,5])/(K[10]+x[...,5])) - (K[49]*(550*x[...,18]*x[...,6] + K[50]*x[...,8] - K[22]*x[...,33]*x[...,6])/NDM[6]))) # IXa        
d8dt = ((TIM*(K[49]*250*x[...,18]*x[...,5] - K[50]*x[...,7]))/NDM[7]) # IX(m)
d9dt = ((TIM*(-K[24]*x[...,22]*x[...,8] - K[26]*x[...,23] + K[49]*550*x[...,18]*x[...,6] - K[50]*x[...,8]))/NDM[8]) # IXa(m)
d10dt = ((TIM*((-K[7]*x[...,4]*x[...,9])/(K[8] + x[...,9]) + K[46]*x[...,11] - K[45]*2700*x[...,18]*x[...,9]))/NDM[9]) # X
d11dt = ((TIM*(K[7]*x[...,4]*x[...,9])/(K[8] + x[...,9]) - K[18]*x[...,31]*x[...,10] + K[19]*x[...,32] - K[21]*x[...,33]*x[...,10] + K[46]*x[...,12] - K[45]*2700*x[...,18]*x[...,10])/NDM[10]) # Xa
d12dt = ((TIM*((-K[27]*x[...,24]*x[...,12])/(K[28] + x[...,11]) - K[46]*x[...,11] - K[45]*2700*x[...,18]*x[...,10]))/NDM[11]) # X(m)
d13dt = ((TIM*((-K[27]*x[...,24]*x[...,12])/(K[28] + x[...,11]) - K[29]*x[...,12]*x[...,27] + K[30]*x[...,28] - K[46]*x[...,12] + K[45]*2700*x[...,18]*x[...,13]))/NDM[12]) # Xa(m)
d14dt = ((TIM*(-K[11]*x[...,10]*x[...,13] - K[48]*x[...,15] - K[47]*2000*x[...,18]*x[...,13]))/NDM[13]) # II
d15dt = ((TIM*(K[11]*x[...,10]*x[...,13] - K[23]*x[...,33]*x[...,14] + K[48]*x[...,16] - K[47]*2000*x[...,18]*x[...,14]))/NDM[14]) # IIa        
d16dt = ((TIM*((-K[31]*x[...,28]*x[...,15])/(K[32] + x[...,15]) - K[48]*x[...,15] - K[47]*2000*x[...,18]*x[...,13]))/NDM[15]) # II(m)
d17dt = (((TIM*(K[31]*x[...,28]*x[...,15])/(K[32] + x[...,15]) - K[48]*x[...,16] - K[47]*2000*x[...,18]*x[...,14]))/NDM[16]) # IIa(m)        
d18dt = (TIM*(((-K[55]*x[...,17]*x[...,18] - K[56]*x[...,17]*x[...,14]))/(1 + x[...,14]))/NDM[17]) # PL
d19dt = (TIM*(((-K[55]*x[...,17]*x[...,18] + K[56]*x[...,17]*x[...,14]))/(1 + x[...,14]))/NDM[18]) # AP
d20dt = (TIM*((-K[12]*x[...,15]*x[...,19])/(K[13] + x[...,19]) - K[51]*750*x[...,18]*x[...,19] + K[52]*x[...,21])/NDM[19]) # VIII
d21dt = (TIM*((K[12]*x[...,15]*x[...,19])/(K[13] + x[...,19]) - K[51]*750*x[...,18]*x[...,20] + K[52]*x[...,22] - K[35]*x[...,20])/NDM[20]) # VIIIa        
d22dt = (TIM*(((K[39]*x[...,16]*x[...,21])/(K[40] + x[...,21])) - ((K[33]*750*x[...,12]*x[...,21])/(K[34] + x[...,21])) + K[51]*750*x[...,18]*x[...,19] - K[52]*x[...,21])/NDM[21]) # VIII(m)        
d23dt = (TIM*(((K[39]*x[...,16]*x[...,21])/(K[40] + x[...,21])) - ((K[33]*750*x[...,12]*x[...,21])/(K[34] + x[...,21])) + K[51]*750*x[...,18]*x[...,20] - K[52]*x[...,21] - K[25]*x[...,22]*x[...,8] + K[26]*x[...,23])/NDM[22]) # VIIIa(m)
d24dt = (TIM*((K[25]*x[...,22]*x[...,8] - K[26]*x[...,23]))/NDM[23]) # IXa:VIIIa
d25dt = (TIM*(((-K[14]*x[...,14]*x[...,24])/(K[15] + x[...,24]) - K[53]*2700*x[...,18]*x[...,24] + K[54]*x[...,26]))/NDM[24]) # V
d26dt = (TIM*(((-K[14]*x[...,14]*x[...,24])/(K[15] + x[...,24]) - K[53]*2700*x[...,18]*x[...,25] + K[54]*x[...,27] - K[38]*x[...,25]))/NDM[25]) # Va
d27dt = (TIM*(((-K[41]*x[...,16]*x[...,26])/(K[42] + x[...,26]) - (K[36]*x[...,12]*x[...,26])/(K[37] + x[...,26]) + K[53]*2700*x[...,18]*x[...,24] - K[54]*x[...,26]))/NDM[26]) # V(m)        
d28dt = (TIM*(((-K[41]*x[...,16]*x[...,26])/(K[42] + x[...,26]) - (K[36]*x[...,12]*x[...,26])/(K[37] + x[...,26]) - K[29]*x[...,12]*x[...,27] + K[30]*x[...,28] + K[53]*2700*x[...,18]*x[...,25] - K[54]*x[...,27]))/NDM[27]) # Va(m)        
d29dt = (TIM*((K[29]*x[...,12]*x[...,27] - K[30]*x[...,28]))/NDM[28]) # Xa:Va
d30dt = (TIM*((-K[16]*x[...,14]*x[...,29])/(K[17] + x[...,29]))/NDM[29]) # I
d31dt = (TIM*((K[16]*x[...,14]*x[...,29])/(K[17] + x[...,29]))/NDM[30]) # Ia
d32dt = (TIM*(-K[18]*x[...,10]*x[...,31] + K[19]*x[...,32])/NDM[31]) # TFPI
d33dt = (TIM*(K[18]*x[...,10]*x[...,31] - K[19]*x[...,32] - K[20]*x[...,4]*x[...,32])/NDM[32]) # xa:TFPI
d34dt = (TIM*(-x[...,33]*(K[21]*x[...,10] + K[22]*x[...,6] + K[23]*x[...,14] + K[24]*x[...,4]) )/NDM[33]) # ATIIddt = 
