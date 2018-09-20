The Python scripts in the paper can be found in `RTS96` folder.
# Local Feature Sufficiency Exploration for Predicting Security-constrained Generation Dispatch in Multi-Area Power Systems
Deriving generation dispatch is essential for efficient
and secure operation of electric power systems. This is usually
achieved by solving a security-constrained optimal power flow
(SCOPF) problem, which is by nature non-convex, usually
nonlinear and thus computationally intensive. The state-of-theart
optimization approaches are not able to solve this problem
for large-scale power systems within power system operation
time window (usually 5 minutes). In this work, we developed
supervised learning approaches to determine security-constrained
generation dispatch within much shorter time window. More
importantly, the physical constraint of only accessing to local
measurements and other information in most utilities’ realtime
operation can not be ignored for the predictive models.
The feasibility and accuracy of utilizing only local features
(measurements and grid information in one area) to predict
optimal local generation dispatch (dispatch of all generators in
the corresponding area) in multi-area power systems has been
explored. The results showed optimal local generation dispatch
can be predicted with local features with high accuracy, which
is comparable to the results obtained with global features.
## Grid Optimization

The objective of GO competition is to accelerate the development of transformational and disruptive methods for solving power system optimization problems, including Preventative Security Constrained AC Optimal Power, where a generation dispatch at the least cost to meet system load in the base case is needed. This project is an attempt to tackle this problem using machine learning regression algorithms.

### Dataset
The dataset employed for the main part of this paper is [`IEEE RTS96`](https://gocompetition.energy.gov/sites/default/files/dataset/Phase_0_RTS96.zip) system, where there are 100 scenarios and 10 contingency conditions. The operational data is stored in `RAW` file and the solution to the generation dispatch is in solution files.

### Data preprocessing
The data preprocessing part consists of 4 steps:
1. Extracting parts between header rows.
2. Concatenating all parts into a longer vector after extracting each part separately.
3. Removing features that have the identical values for all samples.
4. Standardizing data.

For local feature extraction, the implementation of data preprocessing can be found in [`extractArea1.py`](./RTS96/extractArea1.py). The local features from different areas are concatenated in [`feat_imp.py`](./RTS96/feat_imp.py) to generate the global features.

### Random forest
The random forest regression has been adopted for both local data and global data to fit the corresponding generation dispatch. The implementation can be found in [`predictiveModel.py`](./RTS96/predictiveModel.py).

### Feature importance
In order to explore the importance of local features in training process, permutation feature importance([`permutation_imp.py`](./RTS96/permutation_imp.py)) is used in the paper.

### Sobol sensitivity analysis
A global sensitivity analysis is conducted on trained models to find out the contribution of features and their interaction to the model outputs. The implementation can be found in [`sobol_SA01.py`](./RTS96/sobol_SA01.py)
