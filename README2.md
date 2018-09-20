The python scripts from data preprocessing to Sobol sensitivity analysis can be found in `RTS96` folder.


# Grid Optimization

This is an attempt on GO competition dataset using Machine Learning.
The objective of GO competition is to accelerate the development of transformational and disruptive methods for solving power system optimization problems, including Preventative Security Constrained AC Optimal Power, where a generation dispatch at the least cost to meet system load in the base case is needed. This project is an attempt to tackle this problem using machine learning regression algorithms. The initial trial dataset used here is IEEE 14-Bus (100 scenarios).
More details at:

## Data Description
IEEE 14-Bus (100 scenarios) dataset is employed. It contains 100 folders labeled *scenario_1* to *scenario_100*. These scenarios are each independent instances with no coupling to any of the other scenarios. Each scenario folder contains the following files:
* powersystem.raw
* generator.csv
* contingency.csv
* pscopf_data.gms
* pscopf_data.mat
* pscopf.m 
* solution1.txt
* solution2.txt

where only `powersystem.raw` and `solution1.txt` are used in the machine learning models. 

`powersystem.raw` is a PSSE Raw file version 33.5, containing bus, load, fixed shunt, generator branch, transformer, and other control variable datafor a power system. All information needed can be found in this file, however, it may contain other data not relevant to PSCOPF prblem. 

`solution1.txt`contains the required output for the base solution that is produced by the GAMS reference implmentation.

## Data Prepocessing
1. The RAW file contains data from different parts into different rows that are not regularly distributed. The first step is to extract the data of parts into a list. 
This will also take out the white spaces between strings and flatten all the elements to a long vector.
3. Save the first-processed data into a csv file.
4. Clean the data. Since there are many columns sharing the sample value through all the samples, giving the variance equal to 0, which means there is no information can be captured by machine learning algorithms, columns like this are taken out.

## RTS96 with Contingency Data

In order to explore the imporance of local features to local dispatch, a larger system IEEE RTS-96 is employed.  RTS-96 contains 100 different scenaros in which there are RAW file, base solution and contingency solution. The contingency solution file `solution2.txt` is used in this work, where it takes into account 10 different contingencies and the corresponding generation dispatch. Therefore, based on the fact that 10 contingencies are different and independent, the sample size can be expanded to 1000 with each scenario having 10 different contingency situations. 
