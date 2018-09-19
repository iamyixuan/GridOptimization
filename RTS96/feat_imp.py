import pandas as pd
import numpy as np
import os 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

'''
----------------------------
We are going to combine the data 
of 3 areas in order.
----------------------------
'''
area1 = 'cleanedArea1.csv'
area2 = 'cleanedArea2.csv'
area3 = 'cleanedArea3.csv'
dir_path = '/Users/yixuansun/Documents/Research/PNNLrelated/RTS96'


'''
combining data to generate global data
'''

'''area1_data = pd.read_csv(os.path.join(dir_path, area1))

area2_data = pd.read_csv(os.path.join(dir_path, area2))
area3_data = pd.read_csv(os.path.join(dir_path, area3))

# combining data
area12 = np.hstack((area1_data.iloc[:,:-33], area2_data.iloc[:,:-33]))
combined_feat = np.hstack((area12, area3_data.iloc[:,:-33]))

combine_area1 = np.hstack((combined_feat, area1_data.iloc[:,-33:]))
df1 = pd.DataFrame(combine_area1, index = None)
df1.to_csv('Area1Full.csv', index = False)
combine_area2 = np.hstack((combined_feat, area2_data.iloc[:,-33:]))
df2 = pd.DataFrame(combine_area2, index = None)
df2.to_csv('Area2Full.csv', index = False)
combine_area3 = np.hstack((combined_feat, area3_data.iloc[:,-33:]))
df3 = pd.DataFrame(combine_area3, index = None)
df3.to_csv('Area3Full.csv', index = False)'''

'''
------------------------
Explore the feature importance 
to all the targets and then average
the score.
------------------------
'''


# for area 1 target
data = pd.read_csv('Area1Full.csv')
X, y = data.iloc[:,:-33], data.iloc[:,-33:]
# generate a column with random values to explore the reliability of feature importance.
X['Random'] = np.random.random(size = len(X))
color = ['b'] * (len(X.columns)-1) + ['r']
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

reg = RandomForestRegressor()
reg.fit(X, y)

feat_imp = pd.DataFrame({'importance': reg.feature_importances_})
area1_imp = np.mean(feat_imp['importance'].iloc[:167])
area2_imp = np.mean(feat_imp['importance'].iloc[167:334])
area3_imp = np.mean(feat_imp['importance'].iloc[334:])
area_mean_imp = [area1_imp, area2_imp, area3_imp]
print area_mean_imp
#feat_imp.sort_values(by = 'importance', ascending = True, inplace = True)
plt.subplot(121)
plt.barh(range(len(feat_imp)), feat_imp['importance'], 5, color = color, align = 'center')
plt.yticks([0, 167, 334, 501])
plt.xlabel('Feature importance for Area 1')
plt.ylabel('Index')

plt.subplot(122)
plt.barh([1,2,3], area_mean_imp,  color = 'g', align = 'center')
plt.yticks([1, 2, 3])
plt.xlabel('Area Importance Average')
plt.ylabel('Area Number')
plt.tight_layout()
plt.savefig('feat_imp_1.jpg', format = 'jpg', dpi = 300)



# for area 2 targets

data = pd.read_csv('Area2Full.csv')
X, y = data.iloc[:,:-33], data.iloc[:,-33:]
# generate a column with random values to explore the reliability of feature importance.
X['Random'] = np.random.random(size = len(X))
color = ['b'] * (len(X.columns)-1) + ['r']
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
reg = RandomForestRegressor()
reg.fit(X, y)

feat_imp = pd.DataFrame({'importance': reg.feature_importances_})
area1_imp = np.mean(feat_imp['importance'].iloc[:167])
area2_imp = np.mean(feat_imp['importance'].iloc[168:334])
area3_imp = np.mean(feat_imp['importance'].iloc[334:])
area_mean_imp = [area1_imp, area2_imp, area3_imp]
print area_mean_imp
#feat_imp.sort_values(by = 'importance', ascending = True, inplace = True)
plt.subplot(121)
plt.barh(range(len(feat_imp)), feat_imp['importance'], 5, color = color, align = 'center')
plt.yticks([0, 167, 334, 501])
plt.xlabel('Feature importance for Area 2')
plt.ylabel('Index')

plt.subplot(122)
plt.barh([1,2,3], area_mean_imp,  color = 'g', align = 'center')
plt.yticks([1, 2, 3])
plt.xlabel('Area Importance Average')
plt.ylabel('Area Number')
plt.tight_layout()
plt.savefig('feat_imp_2.jpg', format = 'jpg', dpi = 300)

# for area 3

data = pd.read_csv('Area3Full.csv')
X, y = data.iloc[:,:-33], data.iloc[:,-33:]
# generate a column with random values to explore the reliability of feature importance.
X['Random'] = np.random.random(size = len(X))
color = ['b'] * (len(X.columns)-1) + ['r']
reg = RandomForestRegressor()
reg.fit(X, y)


feat_imp = pd.DataFrame({'importance': reg.feature_importances_})
area1_imp = np.mean(feat_imp['importance'].iloc[:167])
area2_imp = np.mean(feat_imp['importance'].iloc[167:334])
area3_imp = np.mean(feat_imp['importance'].iloc[334:])
area_mean_imp = [area1_imp, area2_imp, area3_imp]
print area_mean_imp
#feat_imp.sort_values(by = 'importance', ascending = True, inplace = True)
plt.subplot(121)
plt.barh(range(len(feat_imp)), feat_imp['importance'], 5, color = color, align = 'center')
plt.yticks([0, 167, 334, 501])
plt.xlabel('Feature importance for Area 3')
plt.ylabel('Index')

plt.subplot(122)
plt.barh([1,2,3], area_mean_imp,  color = 'g', align = 'center')
plt.yticks([1, 2, 3])
plt.xlabel('Area Importance Average')
plt.ylabel('Area Number')
plt.tight_layout()
plt.savefig('feat_imp_3.jpg', format = 'jpg', dpi = 300)