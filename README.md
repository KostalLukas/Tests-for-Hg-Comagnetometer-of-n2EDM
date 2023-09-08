# Tests-for-Hg-Comagnetometer-of-n2EDM
### Repository containing the data, analysis and results of the tests carried out for the UV laser system in the Hg co-magnetometer of the n2EDM experiment

#### Organization:
- Each investigation has its own directory in the main repository.
- Each directory contains the Python analysis scripts as well as a Data and Output subdirectories.
- The Data subdrectory contains all of the datasets collected for the particular investigation.
- Data collected using the DAQ system is saved as .txt while data collected manually is saved as .csv
- Large data files have been individually compressed into .xz files
- Different types of tests are specified with a 2 letters as follows:
  Ls - long term stability, St - short test, Ru - power rampup, Rd - power rampdown, Sr - step response
- The Output directory contains an example output generated the Python analysis scipts
- Numerical results are saved as .txt or sometimes .csv files and the plots ae simply saved as .png

#### Investigations:
- Window Test
  - UV window transmission
    - Analysis script is Window Analysis.py
    - Data is saved as Window\_(timestamp)\_W(window no)\_(optional type of test).txt
    - Timestamp has the format mmddhh and denotes the start of data taking
    - Windows are numbered 1-9 as shown in UV_windows.csv
    - Additionally VAC means window has been degassed in a vacuum chamber prior to measurement

- Diode Test
  - Diode comparison
    - Analysis script is Diode Comparison Analysis.py
    - Data is saved as Diode\_D(diode no).txt
    - Diodes are numbered 1-8 as shown in Photo\_diodes.csv
    - Analysis script compares the sensitivity and stability of the photodiodes measured directly at the output of the
      lser without power stabilisation
    - This generates the calibration for each photodiode with the gain of Ch1 on the DAQ
  - Diode tests
    - Analysis script is Diode Analysis.py
    - Data is saved as Diode\_(timestamp)\_D(Ch1 diode no)\_D(Ch2 diode no)\_(type of test).txt
    - Timestamp has the format mmddhh and denotes the start of data taking
    - The analysis script automatically recognizes the photodiode number and loads the calibration and associated error
          from the oputput of Diode Comparison Analysis.py accounting for the different gains on Ch1 and Ch2

- Fiber Test
  - Fiber exposure time
    - Analysis script is Exposure Analysis.py
    - The program goes through all of the DAQ data in the Data subdirectory and compares the time between data files
      detemrined from the timestamp and number of samples in each data file. This is then used to determine the total
      exposure time of the fiber while the DAQ was running
    - The analysis script also saves a list of the final data files as Exposure\_datasets.csv
  - Fiber powermeter measurements
    - Analysis script is PM Analysis.py
    - Data containing measurements taken using th Thorlabs powermeter is saved as PM\_data.csv
    - The analysis program looks at the fiber transmission, beamsplitter ratio and photodiode degradation
    - It also calcualtes an average photodiode calibration constants and associated error
  - Fiber DAQ measurements
    - Analysis script is DAQ Analysis.py
    - Data from DAQ is saved as UV\_new10m\_(date)\_(time).txt be aware some of the data is very large
    - The date and time specify the time at which the data has been downlaoded from the DAQ system
    - The analysis script loads the phodiode calibration constants and associated error calcualted by PM Analysis.py
  - Fiber power ramp
    - Analysis script is Rampup 1 Analysis.py and Rampup 2 Analysis.py
    - Data is saved as Fiber\_(timestamp)\_Ru\_(1 or 2).csv
    - Data no 1 was collected by increasing Vmod to increase power and recoding the DAQ reading and also taking PM measurements
      every 0.5V
      Data no 2 was collected by increasing the power and measuring only the fiber input power using the PM then repeating the
      power increase and measruing only the fiber power output using the PM and repeating this twice  

- Power Stabilisation
  - TA Modulation curves
    - Analysis scipt is Stabilisation Analysis
    - Data is Modulation\_IP.csv and Modulation\_VIP.csv
    - For IP data the TA current Iact is varied manually and recorded while for VIP the TA modulation vltage Vmod is varied
      and both Vmod and Iact are measured in both cases the output power is also measured using the PM
    - For the VIP data the analysis script can take values characterising the current power stabilisation setup and use the PIact curve
      to calcualte parameters for a desired output power

- S130VC Detector Test
  - Sensitivity map
    - Analysis script is S130VC Analysis.py
    - Data is Sensitivity\_map.csv and is a grid of power measured using the Thorlabs S130VC photodode detector with increments of 5 turns on
      a linear translation stage which can be calibrated using the size of the detector window
    - The analysis script plots a heat map characterising the sensitivity across the photodiode active area as well as a standard devaition
      which can be used as the absolute error for all PM measureaments

#### Decompressing the data
- Large datasets are compressed using the LZMA algorith with maximum compression ratio and saved as .xz files
- On Windows the files can be decompressed using the WinZip or 7-Zip utilities
- On Linux or MacOS files can be decompressed using the xz package
1. Install the xz package
2. cd into the Data subdirectory
3. decompress all data with
```console
xz -d -v -T0 *.xz
```
4. set the extension to .txt
```console
for file in *;
  mv -- "$file" "${file%}.txt"
```

#### Running the analysis:
- The analysis Python scripts are designed mainly to be executed from terminal
- The scripts utilise the following packages which have to be installed
  NumPy, matplotlib, SciPy, sys
- The scripts are designed to read process arguments from terminal with which the script was executed

- List of analysis scripts which take arguments:
  - Window Analysis.py: data SPLOT LPF fc
  - Diode Analysis.py: data ACAL SPLOT LPF fc FFT
  - Diode Comparison Analysis.py: P
  - Fiber DAQ Analysis.py: data SPLOT Pth LPF fc
 
- Arguments and their datatypes explained:
  - 'data' - str - filename without extension of data to be analysed in the Data subdirectory
  - 'ACAL' - bool - automatically recognise and calibrate the photodiodes tested
  - SPLOT - bool - subsample measurements to 1000 before plotting to save time and memory
  - LPF - bool - apply a low pass Butterworth filter
  - fc - float or str - cutoff frequency in Hz of the low pass Butterworth filter
  - Pth - float - threshd power in uW if either Ch1 or Ch2 power of measurement is below it is ignored
  - P - float - laser output power in uW at which data for the diode comparison was measured
    
- For boolean arguments use 'True' or 'False' for fc use float or m for fc=1/1min or h for fc=1/1h
- If no arguments are passed the scrypt defaults to preset arguments inside the script 


Last updated: 08.09.2023
By Lukas Kostal
