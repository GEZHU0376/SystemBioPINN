# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 21:42:04 2021

@author: G
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

## CONSTANTS
k1 = 1.0
k2 = 1.0
k3 = 1.0
k4 = 1.0
k5 = 1.0
k6 = 2.3
k7 = 2.3
k10 = 2.0
k12 = 0.52
k13 = 3.8
k18 = 1.0
k19 = 3.0
k20 = 0.4
k21 = 0.4
k22 = 3.0

# FOR K LIST 1
k8_1, k9_1, k11_1, k14_1, k15_1, k16_1, k17_1, k23_1 = 1.28E+00, 5.66E-01, 5.04E-01, 5.10E-01, 1.00E+00, 4.18E-01, 3.23E-03, 2.56E-01

# FOR K LIST 2
k8_2, k9_2, k11_2, k14_2, k15_2, k16_2, k17_2, k23_2 = 5.82E-02, 5.66E-01, 5.01E-01, 5.10E-01, 1.00E+00, 4.60E+00, 4.90E+00, 6.24E-01

# FOR K LIST 3
k8_3, k9_3, k11_3, k14_3, k15_3, k16_3, k17_3, k23_3 = 1.42E-01, 1.80E+00, 5.02E-01, 5.10E-01, 9.98E-01, 1.88E+00, 2.64E+00, 6.29E-01

# FOR K LIST 4
k8_4, k9_4, k11_4, k14_4, k15_4, k16_4, k17_4, k23_4 = 2.93E-02, 1.38E-01, 5.00E-01, 5.10E-01, 9.96E-01, 8.01E+00, 7.47E+00, 7.49E-01

# FOR K LIST 5
k8_5, k9_5, k11_5, k14_5, k15_5, k16_5, k17_5, k23_5 = 1.65E-01, 2.16E+00, 4.99E-01, 5.10E-01, 1.01E+00, 1.58E+00, 2.28E+00, 6.52E-01



k_list_1 =  [k1* 1e6,k2 * 1,k3 * 1,k4 * 1,k5 * 1,
              k6 * 1e-4, k7 * 1e-4,k8_1 * 1,k9_1 * 1e-2,k10 * 1e-6,
              k11_1 *1e-3, k12 * 1e-6, k13 * 1e-8, k14_1 * 1, k15_1 * 1e-3,
              k16_1 * 1e-2, k17_1 * 1e-3, k18 * 1e-3, k19 * 1e-2, k20 * 1,
              k21 * 1, k22 * 1, k23_1 * 1e5]

k_list_2 =  [k1* 1e6,k2 * 1,k3 * 1,k4 * 1,k5 * 1,
              k6 * 1e-4, k7 * 1e-4,k8_2 * 1,k9_2 * 1e-2,k10 * 1e-6,
              k11_2 *1e-3, k12 * 1e-6, k13 * 1e-8, k14_2 * 1, k15_2 * 1e-3,
              k16_2 * 1e-2, k17_2 * 1e-3, k18 * 1e-3, k19 * 1e-2, k20 * 1,
              k21 * 1, k22 * 1, k23_1 * 1e5]

k_list_3 =  [k1* 1e6,k2 * 1,k3 * 1,k4 * 1,k5 * 1,
              k6 * 1e-4, k7 * 1e-4,k8_3 * 1,k9_3 * 1e-2,k10 * 1e-6,
              k11_3 *1e-3, k12 * 1e-6, k13 * 1e-8, k14_3 * 1, k15_3 * 1e-3,
              k16_3 * 1e-2, k17_3 * 1e-3, k18 * 1e-3, k19 * 1e-2, k20 * 1,
              k21 * 1, k22 * 1, k23_3 * 1e5]

k_list_4 =  [k1* 1e6,k2 * 1,k3 * 1,k4 * 1,k5 * 1,
              k6 * 1e-4, k7 * 1e-4,k8_4 * 1,k9_4 * 1e-2,k10 * 1e-6,
              k11_4 *1e-3, k12 * 1e-6, k13 * 1e-8, k14_4 * 1, k15_4 * 1e-3,
              k16_4 * 1e-2, k17_4 * 1e-3, k18 * 1e-3, k19 * 1e-2, k20 * 1,
              k21 * 1, k22 * 1, k23_4 * 1e5]

k_list_5 =  [k1* 1e6,k2 * 1,k3 * 1,k4 * 1,k5 * 1,
              k6 * 1e-4, k7 * 1e-4,k8_5 * 1,k9_5 * 1e-2,k10 * 1e-6,
              k11_5 *1e-3, k12 * 1e-6, k13 * 1e-8, k14_5 * 1, k15_5 * 1e-3,
              k16_5 * 1e-2, k17_5 * 1e-3, k18 * 1e-3, k19 * 1e-2, k20 * 1,
              k21 * 1, k22 * 1, k23_5 * 1e5]

# REAL K LIST
real_k_list = [1.0 * 1e6, 1.0, 1.0, 1.0, 1.0,
              2.3 * 1e-4, 2.3 * 1e-4, 0.4, 2.0 * 1e-2, 2.0 * 1e-6,
              0.5 * 1e-3, 0.52 * 1e-6, 3.8 *1e-8, 0.51, 1.0 * 1e-3,
              0.7 * 1e-2, 2.0 * 1e-3, 1.0 * 1e-3, 3.0 * 1e-2, 0.4,
              0.4, 3.0, 0.6 * 1e5]


k_lists = [k_list_1, k_list_2, k_list_3, k_list_4, k_list_5, real_k_list]
color_list = ['black', 'orange', 'red', 'green', 'cyan', 'blue']
label_list = ["SKP 100s Noiseless - Pred 1", "SKP 100s Noiseless - Pred 2", "SKP 100s Noiseless - Pred 3", "SKP 100s Noiseless - Pred 4", "SKP 100s Noiseless - Pred 5", "Ground Truth"]

def plot():
    name_list = ['SIRPa','CD47.SIRPa','CD47.SIRPa_act','SHP1_inact','SHP1_act',
                 'Myo2a_inact','Myo2a_act','FCR','PSR','FCR_act','PSR_act',
                 'TSP1.CD47.SIRPa','TSP1.CD47.SIRPa_act','CD47_alt','CD47']
    plt.figure(figsize=[4.8 * 5, 4.8 * 3])
    for i in range(len(name_list)):
        plt.subplot(3,5, i+1)
        for z in range(0, 6):
            k_list = k_lists[z]
            
            # Define system

            def model(xvector, t):
                x1 = xvector[0]
                x2 = xvector[1]
                x3 = xvector[2]
                x4 = xvector[3]
                x5 = xvector[4]
                x6 = xvector[5]
                x7 = xvector[6]
                x8 = xvector[7]
                x9 = xvector[8]
                x10 = xvector[9]
                x11 = xvector[10]
                x12 = xvector[11]
                x13 = xvector[12]
                x14 = xvector[13]
                x15 = xvector[14]
                    
                dx1dt = -(k_list[6]*x1*x15 - k_list[20]*x2) - (k_list[2]*k_list[5]*x1*x14 - k_list[2]*k_list[19]*x12)
                dx2dt = (k_list[6]*x1*x15 - k_list[20]*x2) - (k_list[7]*x2*(1-(x5/(x5+k_list[22])))**k_list[21]) + k_list[8]*x3
                dx3dt = (k_list[7]*x2*(1-(x5/(x5+k_list[22])))**k_list[21]) - k_list[8]*x3
                dx4dt = -(k_list[15]*x4*x3) + k_list[16]*x5
                dx5dt = k_list[15]*x4*x3 - k_list[16]*x5
                dx6dt = k_list[12]*x5*x7 - k_list[11]*x6*(x13+x11+x10)
                dx7dt = -k_list[12]*x5*x7 + k_list[11]*x6*(x13+x11+x10)
                dx8dt = -k_list[0]*k_list[1]*k_list[9]*x8 + k_list[10]*x10
                dx9dt = k_list[14]*x11 - k_list[3]*k_list[4]*k_list[13]*x9
                dx10dt = k_list[0]*k_list[1]*k_list[9]*x8 - k_list[10]*x10
                dx11dt = -k_list[14]*x11 + k_list[3]*k_list[4]*k_list[13]*x9
                dx12dt = k_list[2]*k_list[5]*x1*x14 - k_list[2]*k_list[19]*x12 - k_list[17]*x12 + k_list[18]*x13
                dx13dt = k_list[17]*x12 - k_list[18]*x13
                dx14dt = -(k_list[2]*k_list[5]*x1*x14 - k_list[2]*k_list[19]*x12)
                dx15dt = -(k_list[6]*x1*x15 - k_list[20]*x2)
    
                return np.array([dx1dt, dx2dt, dx3dt, dx4dt, dx5dt, dx6dt, dx7dt, dx8dt,
                                 dx9dt, dx10dt, dx11dt, dx12dt, dx13dt, dx14dt, dx15dt])

            # Set the times to obtain a solution

            t = np.linspace(0, 100, 100000)

            # Set initial conditions

            ics = np.array([3000, 0, 0, 2800000, 0, 2200, 0, 50000, 15000,
                            0, 0, 1, 1, 1, 20000])
            
            solution = odeint(model, ics, t)
                
            x1 = solution[:,0]
            x2 = solution[:,1]
            x3 = solution[:,2]
            x4 = solution[:,3]
            x5 = solution[:,4]
            x6 = solution[:,5]
            x7 = solution[:,6]
            x8 = solution[:,7]
            x9 = solution[:,8]
            x10 = solution[:,9]
            x11 = solution[:,10]
            x12 = solution[:,11]
            x13 = solution[:,12]
            x14 = solution[:,13]
            x15 = solution[:,14]
            print((k_list[6]*x1*x15 - k_list[20]*x2), (k_list[7]*x2*(1-(x5/(x5+k_list[22])))**k_list[21]), k_list[8]*x3)
            
            if k_list == real_k_list:
                if i == 0 or i == 3 or i == 4 or i == 7 or i == 8 or i == 9 or i ==  10 or i  == 13 or i == 14:
                    plt.plot(t, solution[:, i], color = color_list[z], marker='o', markersize = 5, markevery = 10000, label="Ground Truth - Data")
                else:
                    plt.plot(t, solution[:, i], color = color_list[z], label = label_list[z])
            else:
                plt.plot(t, solution[:, i], color = color_list[z], label = label_list[z])
            plt.legend()
            plt.title('{} (X{})'.format(name_list[i], i+1), fontsize=18)
            plt.tight_layout()
            
plot()