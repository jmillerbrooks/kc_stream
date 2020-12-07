import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_is_fitted

kc_data = pd.read_csv('./data/kc_house_data.csv')

X, y = kc_data.drop(['price', 'date'], axis=1).values, kc_data['price'].values

# specifying the indices for the columns we'll use below
sqft_living_ind, sqft_living15_ind, bedrooms_ind = 3, 17, 1


# add attributes, a custom sklearn transformer needs init, fit, and transform methods
# fit_transform is created by adding TransformerMixin and BaseEstimator gives get/
# set_params()


class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_sqft_per_bedroom=True,
                 add_sqft_diff=True):
        self.add_sqft_per_bedroom = add_sqft_per_bedroom
        self.add_sqft_diff = add_sqft_diff

    def fit(self, X, y=None):
        # using the helper from sklearn.utils.validation
        # to check the shape of X, y
        X, y = check_X_y(X, y)

        # allows validation through check_is_fitted
        # from sklearn project template example
        self.n_features_ = X.shape[1]

        return self

    def transform(self, X, y=None):

        check_is_fitted(self, 'n_features_')

        # Check input shape, also from sklearn project template
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        if self.add_sqft_per_bedroom:
            # avoid divide by zero error:
            with np.errstate(divide='ignore', invalid='ignore'):
                sqft_per_bedroom = np.true_divide(
                    X[:, sqft_living_ind], X[:, bedrooms_ind])
                sqft_per_bedroom[~np.isfinite(
                    sqft_per_bedroom)] = 0  # -inf inf NaN
            #sqft_per_bedroom = X[:, sqft_living_ind] / X[:, bedrooms_ind]
            #[X, effective_age, sqft_per_bedroom]
        if self.add_sqft_diff:
            sqft_diff = X[:, sqft_living_ind] - X[:, sqft_living15_ind]

        if self.add_sqft_per_bedroom and self.add_sqft_diff:
            X = np.c_[X, sqft_per_bedroom, sqft_diff]
        elif self.add_sqft_per_bedroom:
            X = np.c_[X, sqft_per_bedroom]
        elif self.add_sqft_diff:
            X = np.c_[X, sqft_diff]

        return X
