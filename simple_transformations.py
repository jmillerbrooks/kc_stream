import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_boston


def select_crime_and_taxes(X):
    """ Custom selector for the Boston Housing dataset
    Accepts X, array of 13 features described in sklearn.datasets.load_boston
    Returns only the columns of X corresponding to the features 'CRIM': per capita crime rate, and 'TAX': full-value property-tax rate per $10,000"""
    # select columns in positions 0 and 9, drop remaining columns
    return X[:, [0, 9]]


# we add a simple log scaling transformation
crime_and_taxes_pipe = Pipeline(
    [('feature_selector', FunctionTransformer(select_crime_and_taxes)),
     ('log_scaler', FunctionTransformer(np.log))]
)

if __name__ == "__main__":
    X, y = load_boston(return_X_y=True)
    feature_names = load_boston()['feature_names']
    crime_and_tax_features = crime_and_taxes_pipe.fit_transform(X)
    assert np.all(load_boston()['feature_names'][[0, 9]] == [
                  'CRIM', 'TAX']), 'Feature selection failed'
    plt.style.use('fivethirtyeight')
    plt.hist(crime_and_tax_features[:, 0])
    plt.title("Distribution of Crime Rate after Log Scaling",
              fontsize=14)
    plt.xlabel("Value of 'CRIM'", fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig('./log_scaled_crime.png')
    plt.show()
