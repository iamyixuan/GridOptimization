from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np
import pickle

variable_name = map(str,range(1,502))

'''
define the problem
'''
problem = {
	'num_vars': 501,
	'names': variable_name,
	'bounds':[[-1,1]]*501
}

'''
load the trained model
'''
load_model = pickle.load(open('area3_model.sav', 'rb'))

param_values = saltelli.sample(problem, 50000, calc_second_order=False)
X = param_values
print X.shape
Y = np.mean(load_model.predict(X), axis = 1)
print Y.shape
Si = sobol.analyze(problem, Y, calc_second_order=False, conf_level=0.95, print_to_console=True)
pickle.dump(Si, open('SA_resultsArea3.pkl','wb'))

