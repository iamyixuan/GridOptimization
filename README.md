# Grid Optimization

This is an attempt on GO competition dataset using Machine Learning.
The objective of GO competition is to accelerate the development of transformational and disruptive methods for solving power system optimization problems, including Preventative Security Constrained AC Optimal Power, where a generation dispatch at the least cost to meet system load in the base case is needed. This project is an attempt to tackle this problem using machine learning regression algorithms. The initial trial dataset used here is IEEE 14-Bus (100 scenarios).
More details at:

# Content

* Data Description
* Data Prepocessing
* First Look Into the Data
* Multi-target Regression: Problem transformation
* Neural Networks

## Data Description
IEEE 14-Bus (100 scenarios) dataset is employed. It contrains 100 folders labeled *scenario_1* to *scenario_100*. These scenarios are each independent instances with no coupling to any of the other scenarios. Each scenario folder contrains the following files:
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
1.The RAW file contains data from different parts into different rows that are not regularly distributed. The first step is to extract the data of parts into a list by 
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
There are numerical as well as categorical values in the dataset. The `try except` snippet converts all the numerical values from strings to floats and leaves categorical values strings. Similar procedure can be applied on target values without having to leaving categorical values for there is none.

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

## Multi-target Regression

## Neural Networks
