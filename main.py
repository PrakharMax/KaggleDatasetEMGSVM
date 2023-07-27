import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
from sklearn.model_selection import cross_validate, KFold
from sklearn.svm import SVC
DATASET = 'myo_ds_30l_10ol.npz'
SCORING = ['accuracy', 'f1_macro', 'f1_micro']
data = np.load(DATASET)
X, y = data['X'], data['y']
model = SVC()
results = cross_validate(model, np.median(X, axis=1), y, scoring=SCORING, n_jobs=-1, cv=KFold(5, shuffle=True))
for key, value in results.items():
    print(key)
    print(value)
    print('**********')

