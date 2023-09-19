# -*- coding: utf-8 -*-
"""
ThorLabs Detector S130VC Test Analysis v1.0

Lukas Kostal, 14.7.2023, PSI
"""


import numpy as np
from matplotlib import pyplot as plt
import sys


# function to pass arguments from terminal
def parg(arg_var):
    arg_sys = sys.argv
    
    for i in range(1, len(arg_sys)):
        arg_nam = arg_sys[i].split('=')[0]
        arg_val = arg_sys[i].split('=')[1]
        
        if arg_val == 'm':
            arg_val = 1/60
        if arg_val == 'h':
            arg_val = 1/3600
        
        if arg_nam in arg_var:
            exec(f'{arg_nam} = {arg_val}', globals())
        else:
            print(f'{arg_nam} is not a valid variable')
    return None


# function to print to console and file simultaneously
def tprint(text=''):
    print(text)
    global file
    with open(file, 'a') as output:
        print(text, file=output)


# function to apply threshold
def get_treshold(P, Pth):
    P = np.array(P)
    
    print(P) 
    print()
    print(Pth)
    
    for i in range(0, nx):
        for j in range(0, ny):
            if P[i, j] < Pth:
                P[i, j] = np.nan   
    return P


# function to return average power
def get_avg(P):
    P_avg = np.nansum(P) / np.sum(np.logical_not(np.isnan(P)))
    return P_avg


# function to return standard deviation of power
def get_std(P):
    P_std = np.sqrt(np.nansum((P - get_avg(P))**2) /  np.sum(np.logical_not(np.isnan(P))))
    return P_std


# function to return standard error on the mean of power
def get_sem(P):
    sem = get_std(P) / np.sqrt(np.sum(np.logical_not(np.isnan(P))))
    return sem


# threshold for numerical data and plotting
Pth = 0

# data to be analysed
data = "Sensitivity_map"

# no of turns of adjustement screw on stage between measurements
trn = 2

# distance per revolution of the adjustement screw
dpr = 0.5

# set true to find optimised power threshold
# doesnt work well so default is off
THOPT = True

parg('Pth')

# load data in mW and convert to uW
P = np.loadtxt(f'Data/{data}.csv', unpack=True, delimiter=',', )
P *= 1e3

# get no of measurements in x and y direction respectively
nx, ny = np.size(P, 0), np.size(P, 1)

# get relative position of measurements
x, y = trn * dpr * np.arange(0, nx), trn * dpr * np.arange(0, ny)

# array of thresholds for power in uW with 500 increments
Pth_arr = np.linspace(np.amin(P), np.amax(P), 500)

# array to hold SEM for each threshold
sem_arr = np.zeros(len(Pth_arr))

# loop over all possible thresholds to find one at which SEM is minimum
if THOPT == True:
    for i in range(0, len(Pth_arr)):
        P_now = get_treshold(P, Pth_arr[i])
        sem_arr[i] = get_sem(P_now)

# optimised threshold doesnt really work for now
Pthop = Pth_arr[np.argmin(sem_arr)]
    
# get numerical data
P = get_treshold(P, Pth)
P_avg = get_avg(P)
P_std = get_std(P)
P_sem = get_sem(P)
std_percent = P_std / P_avg * 100
sem_percent = P_sem / P_avg * 100

# print the numerical results
file = f'Output/results_{Pth}.txt'
open(file,'w')

tprint(f'powe threshold       = {Pth:.4g} uW')
tprint(f'mean power           = {P_avg:.4g} uW')
tprint(f'std power            = {P_std:.4g} uW')
tprint(f'SEM power            = {P_sem:.4g} uW')
tprint(f'STD percentage       = {std_percent:.4g} %')
tprint(f'SEM percentage       = {sem_percent:.4g} %')

# parameters for plotting heat map with optimsied threshold
plt.figure(1)
plt.title('Sensitivity Map of S130VC Detector Area')
plt.xlabel('horizontal position $x$ (mm)')
plt.ylabel('vertical position $y$ (mm)')
plt.gca().set_aspect('equal')
plt.xticks(x)
plt.yticks(y)

# plot heat map and colorbar
plt.pcolormesh(P, cmap='inferno')
plt.colorbar(label='measured power $P$ ($\mu W$)')

# save plot
plt.savefig(f'Output/heatmap_{Pth}.png', dpi=300, bbox_inches='tight')

# if optimised piower threshold show SEM against threshold
if THOPT == True:
    
    # parameters for plotting SEM against threshold
    plt.figure(2)
    plt.title('Standard Error on the Mean against Threshold')
    plt.xlabel('threshold $P_{th}$ ($\mu W$)')
    plt.ylabel('SEM $\delta_P$ ($\mu W$)')
    plt.rc('grid', linestyle=':', color='black', alpha=0.8)
    plt.grid()
    
    # plot standard error on the mean
    plt.plot(Pth_arr, sem_arr, color='royalblue')
    
    # save plot
    plt.savefig('Output/optimisation.png', dpi=300, bbox_inches='tight')

# show plots
plt.show()