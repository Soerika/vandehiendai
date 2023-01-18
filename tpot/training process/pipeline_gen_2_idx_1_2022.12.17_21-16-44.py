import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=173808)

# Average CV score on the training set was: -88956.98985191065
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.95, learning_rate=0.001, loss="quantile", max_depth=8, max_features=0.1, min_samples_leaf=19, min_samples_split=12, n_estimators=100, subsample=0.9000000000000001)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=1, min_samples_split=2, n_estimators=100)),
    AdaBoostRegressor(learning_rate=0.5, loss="square", n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 173808)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
