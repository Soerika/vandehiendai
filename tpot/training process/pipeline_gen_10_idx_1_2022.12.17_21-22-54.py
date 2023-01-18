import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=173808)

# Average CV score on the training set was: -77809.85211149162
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.25, min_samples_leaf=1, min_samples_split=2, n_estimators=100))
    ),
    ExtraTreesRegressor(bootstrap=False, max_features=0.8500000000000001, min_samples_leaf=1, min_samples_split=10, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 173808)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
