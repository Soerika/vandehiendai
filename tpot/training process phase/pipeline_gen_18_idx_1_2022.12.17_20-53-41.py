import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=173808)

# Average CV score on the training set was: -71168.21608775205
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    GradientBoostingRegressor(alpha=0.9, learning_rate=0.1, loss="ls", max_depth=5, max_features=0.9500000000000001, min_samples_leaf=7, min_samples_split=14, n_estimators=100, subsample=0.9500000000000001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 173808)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
