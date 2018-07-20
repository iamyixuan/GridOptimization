# Grid Optimization

This is an attempt on GO competition dataset using Machine Learning.
The objective of GO competition is to accelerate the development of transformational and disruptive methods for solving power system optimization problems, including Preventative Security Constrained AC Optimal Power, where a generation dispatch at the least cost to meet system load in the base case is needed. This project is an attempt to tackle this problem using machine learning regression algorithms. The initial trial dataset used here is IEEE 14-Bus (100 scenarios).
More details at:

# Content

* [Data Description](#data-description)
* [Data Prepocessing](#data-prepocessing)
* [First Look Into the Data](#first-look-into-the-data)
* [Multi-target Regression: Problem transformation](#multi-target-regression)
* [Neural Networks](#neural-networks)
* [Some Modification in Progress](#some-modifications-in-progress)
* [RTS96 with Contingency Data](#rts96-with-contingency-data)

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
1. The RAW file contains data from different parts into different rows that are not regularly distributed. The first step is to extract the data of parts into a list by 
```python
def extractCertainLines(startLine, endLine, scenarioNum):
	dir_path = '/home/yixuan/Downloads/Phase_0_IEEE14'
	dir_path = os.path.join(dir_path, scenarioNum)
	txt_file = 'powersystem.raw'
	features = []
	with open(os.path.join(dir_path, txt_file)) as file:
		for i, line in enumerate(file):
			if i <= endLine - 1 and i >= startLine - 1:
				newLine = line.strip().replace(' ','').replace("'","").split(',')
				features.append(newLine)
	oneDlist = [item for sublist in features for item in sublist]
	return oneDlist
  ```
This will also take out the white spaces between strings and flatten all the elements to a long vector. Same operation can be done for `solution1.txt` to extract the base solution as a 10-dimesional vector each sample.

2. Concatenate all parts into a longer vector after extracting each part separately. 
```python
def makeFeautures(scenarioNum):
	bus_data = extractCertainLines(4, 17, scenarioNum)
	load_data = extractCertainLines(19, 29, scenarioNum)
	fixed_shunt_data = extractCertainLines(31, 31, scenarioNum)
	generator_data = extractCertainLines(33, 37, scenarioNum)
	branch_data = extractCertainLines(39, 55, scenarioNum)
	transformer_data = extractCertainLines(57, 68,scenarioNum)
	feat = [bus_data, load_data, fixed_shunt_data, generator_data,
			branch_data, transformer_data]
	sampleFeatures = [item for sub in feat for item in sub]
	for i, x in enumerate(sampleFeatures):
		try:
			sampleFeatures[i] = float(x)
		except ValueError:
			pass
	return sampleFeatures
  ```
There are numerical as well as categorical values in the dataset. The `try except` snippet converts all the numerical values from strings to floats and leaves categorical values strings. Similar procedure can be applied on target values without having to leave categorical values for there is not such values.

3. Save the first-processed data into a csv file.
```python
dataset = []
for names in filenames:
	sample = combineTogether(names)
	dataset.append(sample)
dataframe = pd.DataFrame(dataset)
dataframe.to_csv('integratedDataset.csv')
```
Now the saved dataset consists of 100 rows (100 scenarios aforementioned) and 886 columns (876 features and 10 targets per sample).

4. Clean the data. Since there are many columns sharing the sample value through all the samples, giving the variance equal to 0, which means there is no information can be captured by machine learning algorithms, columns like this are taken out.
```python
def cleanData(filename):
	data = pd.read_csv(filename, header = 0)
	data = data.iloc[:,1:]
	data = data.dropna(axis = 1)
	cols = data.columns
	num_cols = data._get_numeric_data().columns 
	catData = list(set(cols) - set(num_cols))# detecting categorical features.
	data = data.drop(catData, axis = 1) # dropping all categorical features because they are the same for all samples.
	nunique = data.apply(pd.Series.nunique) # find out the repeat data.
	colsToDrop = nunique[nunique == 1].index
	data = data.drop(colsToDrop, axis = 1)# drop out the columns containing the same value.
	columnsNmae = data.columns # store the column name for later indexing
	return data, columnsNmae
  ```
  The data after cleaning consists of 100 rows and 52 columns (42 features and 10 targets).
## First Look Into the Data
Due to the limition of sample size and the nature of a multi-target regression problem, it is important to learn the correlation between features, targets as well as features and targets. If two features have strong correlation, taking out one of them could be a good way to reduce the dimension, preventing overfitting. If two or more targets are strongly correlated, one should consider algorithm adaptation methods to capture the relations between targets. 
```python
dataset, colName = cleanData(filename)
X = dataset.iloc[:,:42]
y = dataset.iloc[:,42:]
plt.matshow(dataset.corr())
plt.savefig("correlation.jpg", dpi = 1000, format = 'jpg')
```
![alt text](https://github.com/sunyx1223/GridOptimization/blob/master/correlation.jpg)
The correlation matrix does not show any strong correlations between features, targets or between features and targets. Thus, further feature selection techniques have not been employed so far, and the correlation between features indicates problem transformation methods may be feasible, where a muilt-target regression problem is treated as several independent single-target problems. 
## Multi-target Regression
The problem transformation is achieved by using `sklearn.multioutput.MultiOutputRegressor` , where the user choose an algorithm for single-target regression problem and `MultiOutputRegressor` puts the results of sub-problems together and evaluates them. The machine learning algorithms for single-target are:
* `LinearRegression`
* `RandomForest`
* `RidgeRegression`
* `LassoRegression`
* `SVR`
* `BaysianRidge`

Unfortunately, all of the models show severe overfitting. The following table shows the results. 

| Algorithms | Coefficient of determination |
| ---------- |:--------------------------:|
| Linear Regression | -0.2148 |
| Random Forest | -0.2148 |
| Ridge Regression|  -1.066 |
| Lasso Regression | -0.1698 |
| Support Vector Regression (RBF) | -0.0963 |
| Baysian Ridge Regression | -0.1049 |

The coefficients of determination of testing set from the models in the table are all negative, indicating the trained models cannot make reasonable predictions on unseen data at all. Therefore, the data may contain complex non-linearity between features and targets that needs a deep model to explore.

## Neural Networks
A fully connected deep neural network is built for this problem, since it is a multi-target regression where the targets do not have much of a correlation with each other. The input layer consists of 42 nodes as there are 42 features. The output layer contains 10 nodes representing 10 outputs for each sample. In between, there are 4 hidden layers that contain 1000, 800, 500, 200 nodes, respectively. 

The network is constructed using Tensorflow, a deep learning framework.
1. Create placeholders for data to feed in and batch data from Dataset class.
```python
X = tf.placeholder(tf.float32, shape = (None, n_input), name = 'X')
y = tf.placeholder(tf.float32, shape = (None, n_output), name = 'y')
batch_size = tf.placeholder(tf.int64)

data = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).repeat()
iter = data.make_initializable_iterator()
features, labels = iter.get_next()
```
2. Define the arithmetic between fully connected layers
```python
def neruon_layer(X, n_neurons, name, activition = None):
	with tf.name_scope(name):
		n_input = int(X.get_shape()[1])
		stddev = 2/np.sqrt(n_input)
		init = tf.truncated_normal((n_input, n_neurons), stddev = stddev)
		W = tf.Variable(init, name = 'kernel')
		b = tf.Variable(tf.zeros([n_neurons]), name = 'bias')
		Z = tf.matmul(X,W) + b
		if activition is not None:
			return activition(Z)
		else:
			return Z
```
3. Construct the body of the network
```python
with tf.name_scope('dnn'):
	hidden1 = neruon_layer(features, n_hidden1, name = 'hidden1', activition = tf.nn.relu)
	dropout = tf.nn.dropout(hidden1, keep_prob = 0.5)
	hidden2 = neruon_layer(dropout, n_hidden2, name = 'hidden2', activition = tf.nn.relu)
	dropout2 = tf.nn.dropout(hidden2, keep_prob = 0.5)
	hidden3 = neruon_layer(dropout2, n_hidden3, name = 'hidden3', activition = tf.nn.relu)
	dropout3 = tf.nn.dropout(hidden3, keep_prob = 0.5)
	hidden4 = neruon_layer(dropout3, n_hidden4, name = 'hidden4', activition = tf.nn.relu)
	dropout4 = tf.nn.dropout(hidden4, keep_prob = 0.5)
	output = neruon_layer(dropout, n_output, name = 'output', activition = None)
```
4. Specify the loss function
```python
with tf.name_scope('loss'):
	mse = tf.losses.mean_squared_error(labels = labels, predictions = output)
```
5. Training operation (minimizing the loss)
```python
with tf.name_scope('train'):
	optimizer = tf.train.AdamOptimizer()
	training_op = optimizer.minimize(mse)
```

6. Execution 
```python
initializer = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(initializer)
	for epoch in range(n_epoch):
		sess.run(iter.initializer, feed_dict = {X: X_train, y: y_train, batch_size: BATCH_SIZE})
		train_error = []
		for i in range(num_batches):
			sess.run(training_op)
			train_error.append(mse.eval())
			
		sess.run(iter.initializer, feed_dict = {X: X_test, y: y_test, batch_size: len(y_test)})
		acc_test = acc.eval()
		print(epoch + 1, "Train loss", np.mean(train_error), "Test Loss: %.4f" % acc_test)
```
In the training process, mean squared error loss function is employed and Adam optimizer is used. The initial training (without dropout layers) leads to a severe overfiting, where

`Train loss: 2.8762033 & Test loss: 1584.9363`

In order to address the overfitting problem, dropout layers are used between fully connected layers with a dropout rate as 0.5. This solves the overfitting problem, however; the model is not able to learn much from the data for the training loss stops converging after 50 epochs. 
```
Train loss: 1013.13104, Test Loss: 1026.6914
```
The training loss and test loss would just oscillate around these numbers for more epochs. The best results show up at epoch 
28, where
```
Train loss: 929.6456, Test Loss: 936.4531
```
This model is still not what is wanted, since it still can not fit the data well enough. Notice that there are only 80 samples used in training and 20 in testing (100 data points in total), which it's too small for a dataset that has 42 features. 

The next step done is to use the boostrapping resampling technique to resample data points from the empirical distribution. The sample size is increased to 1000 after resampling. Although the resampled data does not make any sense to the real world, the predictive power of the model is drastically improved.
```
Train loss: 44.991547, Test Loss: 51.1006
```

The major problem encountered at this stage, from my perspective, is the limited sample size. It is not big enough for the models to learn the pattern well. 
## Some modifications in progress
### Features used in the models
1. Bus data is the same for all samples, so it is taken out.
2. Load data: column 6 & 7. (11 * 2 = 22 starting from 2 to 14)
3. Fixed shunt data is the same for all samples, so it is taken out.
4. Generator data: column 5 & 6 & 18 & 19 (5 * 4 = 20).
5. Branch data is the same for all samples, so it is taken out.
6. Transformer data is the same for all samples, so it is taken out.

Other data categories contain few data, so they are not taken into consideration.
### More generalized code for data extraction
```python
def findlines(scenarioNum):
	dir_path = '/home/yixuan/Downloads/Phase_0_IEEE14'
	dir_path = os.path.join(dir_path, scenarioNum)
	txt_file = 'powersystem.raw'
	headerRow = []
	startRow = []
	endRow = []
	with open(os.path.join(dir_path, txt_file)) as file:
		for num, line in enumerate(file):
			line = line.strip()
			if '/' in line:
				headerRow.append(num)
		for start, end in zip(headerRow[:-1], headerRow[1:]):
			startRow.append(start)
			endRow.append(end)
	return	list(np.array(startRow) +1), list(np.array(endRow) - 1)
```
This function is able to identify the rows where the headers are and return the rows in between, where the data is stored.
And in turn, the data combining process becomes the following:
```python
def integrateFeat(scenarioNum):
	startLine, endLine = findlines(scenarioNum)
	feat = []
	for i in range(len(startLine)):
		feat.append(extractCertainLines(startLine[i], endLine[i], scenarioNum))
	sampleFeatures = [item for sub in feat for item in sub]
	for i, x in enumerate(sampleFeatures):
		try:
			sampleFeatures[i] = float(x)
		except ValueError:
			pass
	return sampleFeatures
```
### Performed feature importance visualization based on Random Forest
Bagging tree based algorithms like Random Forest can be used to investigate feature importance. In this project, considering there are 10 targets, the investigation is conducted for each features-target pair. Feature importances vary with respect to different targets. Here are the feature importance plots

For target #1
![alt text](https://github.com/sunyx1223/GridOptimization/blob/master/FeatureImportanceWithNames/feat_imp1.jpg)
For target #2
![alt text](https://github.com/sunyx1223/GridOptimization/blob/master/FeatureImportanceWithNames/feat_imp2.jpg)
For target #3
![alt text](https://github.com/sunyx1223/GridOptimization/blob/master/FeatureImportanceWithNames/feat_imp3.jpg)
For target #4
![alt text](https://github.com/sunyx1223/GridOptimization/blob/master/FeatureImportanceWithNames/feat_imp4.jpg)
For target #5
![alt text](https://github.com/sunyx1223/GridOptimization/blob/master/FeatureImportanceWithNames/feat_imp5.jpg)
For target #6
![alt text](https://github.com/sunyx1223/GridOptimization/blob/master/FeatureImportanceWithNames/feat_imp6.jpg)
For target #7
![alt text](https://github.com/sunyx1223/GridOptimization/blob/master/FeatureImportanceWithNames/feat_imp7.jpg)
For target #8
![alt text](https://github.com/sunyx1223/GridOptimization/blob/master/FeatureImportanceWithNames/feat_imp8.jpg)
For target #9
![alt text](https://github.com/sunyx1223/GridOptimization/blob/master/FeatureImportanceWithNames/feat_imp9.jpg)
For target #10
![alt text](https://github.com/sunyx1223/GridOptimization/blob/master/FeatureImportanceWithNames/feat_imp10.jpg)

### Looking into the correlation between generator data and genenration dispatch data.

Generator data consists of `QMax`, `QMin`, `PMax`, `PMin` for each bus, while generation dispatch data only contains `Q` and `P`. The scatter plots are to visulize the possible correlation between them. For each bus, 4 scatter plots are made. They are `PMax - P`, `PMin - P`, `QMax - Q` and `QMin - Q`. 

Bus 01
![alt text](https://github.com/sunyx1223/GridOptimization/blob/master/scatterplots/BUS01_scatter.jpg)

Bus 02
![alt text](https://github.com/sunyx1223/GridOptimization/blob/master/scatterplots/BUS02_scatter.jpg)

Bus 03
![alt text](https://github.com/sunyx1223/GridOptimization/blob/master/scatterplots/BUS03_scatter.jpg)

Bus 06
![alt text](https://github.com/sunyx1223/GridOptimization/blob/master/scatterplots/BUS06_scatter.jpg)

Bus 08
![alt text](https://github.com/sunyx1223/GridOptimization/blob/master/scatterplots/BUS08_scatter.jpg)

The plots do not indicate obvious correlation between the variables.

## RTS96 with Contingency Data

In order to explore the imporance of local features to local dispatch, a larger system IEEE RTS-96 is employed. Similar to the previous dataset (IEEE 14), RTS-96 contains 100 different scenaros in which there are RAW file, base solution and contingency solution. The contingency solution file `solution2.txt` is used in this work, where it takes into account 10 different contingencies and the corresponding generation dispatch. Therefore, based on the fact that 10 contingencies are different and independent, the sample size can be expanded to 1000 with each scenario having 10 different contingency situations. 

RTS-96 system consists of three areas where the bus ID has the form 1xx, 2xx and 3xx. The alignment is shown in the following image.
![alt_text](https://github.com/sunyx1223/GridOptimization/blob/master/scatterplots/1-s2.0-S221053791400078X-gr3.jpg)

Our objective is to explore whether local features, e.g. system operational data in area 1 with contingency information, is enough for local dispatch (dispatch for area 1 in this example). According to this requirement, the information extraction from the RAW file can be divided into two:

1. Only extracting local features and contingency data combined with local dispatch.
2. Extracting all features of 3 areas and contingency data combined with local dispatch.

There are 6 file generated, 3 for each type of dataset. 

### Predictive Model 

The first attempt is to use local features only to fit predictive models. Same as the previous model on IEEE 14, Random Forest regressor with Muilt-output regressor is used. The following table shows the results.

| Area Number | Coefficient of determination (Local Data) | Coefficient of determination (Global Data)|
| ---------- |:--------------------------:|----|
| 1 | 0.9751 | 0.8716 |
| 2 | 0.8688 | 0.9763 |
| 3 | 0.9743 | 0.9747 |

Area 1 & 3 show pretty good predictive power just using local features while its local features for area 2 seem not to be enough. If looking back at the alignment map of RTS-96, area 2 is connected with both area 1 & 3, while area 1 and area 3 are relatively independent. An assumption is that the dispatch of area 2 is more dependent on information other than its local data. 

If using all information available to predict local generation dispatch, the results in the table show that for area 1, gloabl information may be redundant for the score decreases, however, the global information drastically helps improve the model predictive power for area 2. The gobal information has little impact on the dispatch prediction in area 3.
