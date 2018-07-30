import pandas as pd
import numpy as np
import os 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
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


area1_data = pd.read_csv(os.path.join(dir_path, area1))
area2_data = pd.read_csv(os.path.join(dir_path, area2))
area3_data = pd.read_csv(os.path.join(dir_path, area3))

# combining data
area12 = np.hstack((area1_data.iloc[:,:-33], area2_data.iloc[:,:-33]))
combined_feat = np.hstack((area12, area3_data.iloc[:,:-33]))

combine_area1 = np.hstack((combined_feat, area1_data.iloc[:,-33:]))
combine_area2 = np.hstack((combined_feat, area2_data.iloc[:,-33:]))
combine_area3 = np.hstack((combined_feat, area3_data.iloc[:,-33:]))

'''
------------------------
Explore the feature importance 
to all the targets and then average
the score.
------------------------
'''
def permutation_importances(rf, X_train, y_train, metric):
    baseline = metric(rf, X_train, y_train)
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = metric(rf, X_train, y_train)
        X_train[col] = save
        imp.append(baseline - m)
    return np.array(imp)


def oob_regression_r2_score(rf, X_train, y_train):
    """
    Compute out-of-bag (OOB) R^2 for a scikit-learn random forest
    regressor. We learned the guts of scikit's RF from the BSD licensed
    code:
    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L702
    """
    X = X_train.values
    y = y_train.values

    n_samples = len(X)
    predictions = np.zeros(n_samples)
    n_predictions = np.zeros(n_samples)
    for tree in rf.estimators_:
        unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples)
        tree_preds = tree.predict(X[unsampled_indices, :])
        predictions[unsampled_indices] += tree_preds
        n_predictions[unsampled_indices] += 1

    if (n_predictions == 0).any():
        warnings.warn("Too few trees; some variables do not have OOB scores.")
        n_predictions[n_predictions == 0] = 1

    predictions /= n_predictions

    oob_score = r2_score(y, predictions)
    return oob_score



# for area 1 target

X, y = combine_area1[:,:-33], combine_area1[:,-33:]
reg = RandomForestRegressor()
reg.fit(X, y)

feat_imp = pd.DataFrame({'importance': reg.feature_importances_})
area1_imp = np.mean(feat_imp['importance'].iloc[:168])
area2_imp = np.mean(feat_imp['importance'].iloc[168:336])
area3_imp = np.mean(feat_imp['importance'].iloc[336:])
area_mean_imp = [area1_imp, area2_imp, area3_imp]
print area_mean_imp
#feat_imp.sort_values(by = 'importance', ascending = True, inplace = True)
plt.subplot(121)
plt.barh(range(len(feat_imp)), feat_imp['importance'], 5, color = 'b', align = 'center')
plt.yticks([0, 168, 336, 500])
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

'''X, y = combine_area2[:,:-33], combine_area2[:,-33:]
reg = RandomForestRegressor()
reg.fit(X, y)

feat_imp = pd.DataFrame({'importance': reg.feature_importances_})
area1_imp = np.mean(feat_imp['importance'].iloc[:168])
area2_imp = np.mean(feat_imp['importance'].iloc[168:336])
area3_imp = np.mean(feat_imp['importance'].iloc[336:])
area_mean_imp = [area1_imp, area2_imp, area3_imp]
print area_mean_imp
#feat_imp.sort_values(by = 'importance', ascending = True, inplace = True)
plt.subplot(121)
plt.barh(range(len(feat_imp)), feat_imp['importance'], 5, color = 'b', align = 'center')
plt.yticks([0, 168, 336, 500])
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

X, y = combine_area3[:,:-33], combine_area3[:,-33:]
reg = RandomForestRegressor()
reg.fit(X, y)


feat_imp = pd.DataFrame({'importance': reg.feature_importances_})
area1_imp = np.mean(feat_imp['importance'].iloc[:168])
area2_imp = np.mean(feat_imp['importance'].iloc[168:336])
area3_imp = np.mean(feat_imp['importance'].iloc[336:])
area_mean_imp = [area1_imp, area2_imp, area3_imp]
print area_mean_imp
#feat_imp.sort_values(by = 'importance', ascending = True, inplace = True)
plt.subplot(121)
plt.barh(range(len(feat_imp)), feat_imp['importance'], 5, color = 'b', align = 'center')
plt.yticks([0, 168, 336, 500])
plt.xlabel('Feature importance for Area 3')
plt.ylabel('Index')

plt.subplot(122)
plt.barh([1,2,3], area_mean_imp,  color = 'g', align = 'center')
plt.yticks([1, 2, 3])
plt.xlabel('Area Importance Average')
plt.ylabel('Area Number')
plt.tight_layout()
plt.savefig('feat_imp_3.jpg', format = 'jpg', dpi = 300)



