import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=173808)

# Average CV score on the training set was: -72476.32760184366
exported_pipeline = GradientBoostingRegressor(alpha=0.8, learning_rate=0.1, loss="ls", max_depth=10, max_features=0.45, min_samples_leaf=3, min_samples_split=17, n_estimators=100, subsample=0.6500000000000001)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 173808)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
