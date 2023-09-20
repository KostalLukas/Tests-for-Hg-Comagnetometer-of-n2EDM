# -*- coding: utf-8 -*-
"""
Fiber Exposure Counter v1.1

Lukas Kostal, 4.9.2023, PSI
"""


import numpy as np
from matplotlib import pyplot as plt
import glob as gb
import sys


# function to pass arguments from console
def parg(*arg_var):
    arg_sys = sys.argv[1:]
    
    arg_name = []
    arg_type = []
    for i in range(0, len(arg_var)):
        arg_id = id(arg_var[i])
        
        for key in globals().keys():
            if key[0] != '_':
                val = globals()[key]
                if id(val) == arg_id:
                    arg_name.append(key)
                    arg_type.append(type(val))
                
    for i in range(0, len(arg_sys)):
        for j in range(0, len(arg_var)):
            if arg_sys[i].split('=')[0] == arg_name[j]:
                
                arg_val = arg_sys[i].split('=')[1]
                
                if arg_val == 'm':
                    arg_val = 1/60
                if arg_val == 'h':
                    arg_val = 1/3600
                
                globals()[arg_name[j]] = arg_type[j](arg_val)
    return None

    
# function to print to console and file simultaneously
def tprint(text=''):
    print(text)
    global file
    with open(file, 'a') as output:
        print(text, file=output)
        
        
# threshold for time difference at which elapsed time is added to exposure time
dt_th = 2

# sampling freuqnecy in Hz
fs  = 10

# specify no of days in each month
# taking 2024 to be a leap year otherwise change 29 -> 28
mtd = np.array([31, 30, 29, 31, 30, 31, 30, 31, 30, 31, 30, 31])

# get threshold for time difference
parg(dt_th)

# get a sorted list of all datasets in the Data directory
ds = gb.glob('Data/UV_new10m_**')
ds.sort()

# no of all datasets
n_ds = len(ds)

# remove all datasets which have additional information in the name
i = 0
while i < n_ds:
    if len(ds[i]) > 35:
        ds.remove(ds[i])
        n_ds -= 1
    else:
        i += 1
        
# array to hold the date when a dataset was doawnloaded from the DAQ     
date = np.zeros((4, n_ds))

# array to hold the number of samples in a dataset
n_samp = np.zeros(n_ds)

# loop over all datsets to get the date and no of samples
for i in range(0, n_ds):
    # get the date from the filename
    date[0, i] = int(ds[i][15:17])
    date[1, i] = int(ds[i][17:19])
    date[2, i] = int(ds[i][20:22])
    date[3, i] = int(ds[i][23:25])
    
    # get the no of measurements from the no of lines
    with open(ds[i], "rbU") as file:
        n_samp[i] = sum(1 for _ in file)
    
    i_ds = np.char.zfill(str(i +1), int(np.ceil(np.log10(n_ds +2))))
    print(f'Reading dataset {i_ds}/{n_ds+1} \t {ds[i][5:]}')

print()

# elapsed time determined from no of samples in datasets in h
t_samp = n_samp / fs / 3600

# arrays of time elapsed between two consecutive datasets
et_date = np.zeros(n_ds)
et_samp = np.zeros(n_ds)

for i in range(1, n_ds):
    # convert the difference in dates to elapsed time in h
    et_date[i] += (date[0, i] - date[0, i-1]) * mtd[int(date[0, i] - 1)] * 24
    et_date[i] += (date[1, i] - date[1, i-1]) * 24
    et_date[i] += (date[2, i] - date[2, i-1])
    et_date[i] += (date[3, i] - date[3, i-1]) / 60
    
    # calcualte the time elapsed between datasets from no of samples in h
    et_samp[i] = t_samp[i] - t_samp[i-1]

# differences in elapsed times calcualted from date and no of samples 
dt_expo = et_date - et_samp

# final time for which the fiber has been exposed while the DAQ was sampling
t_expo = 0
i_expo = []
for i in range(0, n_ds):
    # if difference in elapsed time is significantly negative raise an error
    if dt_expo[i] < -1:
       raise Warning('Dataset contains more data than could have been possibly connected.')
    
    # if difference in elapsed time is grater than the set threshold include
    # the time from the last dataset to the total exposure time
    if dt_expo[i] > dt_th:
        t_expo += t_samp[i-1]
        i_expo.append(i)
        
i_expo = np.array(i_expo)

# write the datasets used to a csv file
with open('Output/Exposure_datasets.csv', 'w') as datasets:
    for i in range(0, len(i_expo)):
        print(ds[i_expo[i] -1], file=datasets)

# print the numerical results
file = 'Output/Exposure_results.txt'
open(file, 'w')
    
tprint(f'total exposure time      = {t_expo:.2f} h')
tprint()
tprint(f'no of datasets           = {n_ds}' )
tprint(f'start date               = {date[0, 0]}.{date[1, 0]}.{date[2, 0]}.{date[3, 0]}')
tprint(f'end date                 = {date[0, -1]}.{date[1, -1]}.{date[2, -1]}.{date[3, -1]}')
tprint(f'total datasets timespan  = {np.sum(t_samp):.2f} h')
tprint()
tprint(f'dt threshold             = {dt_th} h')


# parameters for plotting sampled time from datasets 
plt.figure(1)
plt.title('Sampled Time from Datasets')
plt.xlabel('index $i$ (unitless)')
plt.ylabel('sampled time $t_s$ (h)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(t_samp, c='royalblue')
plt.plot(i_expo-1, t_samp[i_expo-1], 'x', c='red')

# parameters for plotting difference in elapsed times
plt.figure(2)
plt.title('Difference in Elapsed Time Between Datasets')
plt.xlabel('index $i$ (unitless)')
plt.ylabel('time difference $\Delta t$ (h)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(dt_expo, c='royalblue')
plt.axhline(y=dt_th, ls='--', c='red')
plt.plot(i_expo, dt_expo[i_expo], 'x', c='red')

plt.show()

# The program uses datasets to determine the total time for which the fiber has
# been exposed while the DAQ was sampling. This is done by calculating the time
# elapsed between datasets from the date in their filenames and comparing it with
# the time elapsed between measurements during which the DAQ was sampling calculated
# from the difference in no of samples in consecutive datasets. From this it can
# be determined if the DAQ has been stopped and therefore weather the time from
# the dataset should be included in the total fiber exposure time. 

    

        

        
