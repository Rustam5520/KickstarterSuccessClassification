import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
# Loading initial data
data_raw = pd.read_csv('Data\ks-projects-201801.csv')

# Checking initial target values
data_raw.state.unique()
# print(data_raw.columns)

# Leaving only desired classes for binary classification
desired_classes = ['failed', 'successful']
data_raw = data_raw[data_raw.state.isin(desired_classes)]
# data_raw.state.unique()

# Dealing with dates and deriving new variable: 'period'
data_raw.launched = pd.to_datetime(data_raw.launched)
data_raw.deadline = pd.to_datetime(data_raw.deadline)
data_raw['period'] = data_raw.deadline.subtract(data_raw.launched)
data_raw.period = data_raw.period.astype('timedelta64[D]')

# Leaving predictor that will be used to train models and separate target
final_cols = ['goal', 'backers', 'period', 'category', 'main_category', 'currency', 'state']
data_raw = data_raw[final_cols]
Y_values = pd.get_dummies(data_raw.pop('state'))['successful']

# Instantiate encoder/scaler
columns_to_encode = ['category', 'main_category', 'currency']
columns_not_to_encode = ['goal', 'backers', 'period']

ohe = OneHotEncoder(sparse=False)
ohe.fit(data_raw[columns_to_encode])

encoded_columns = ohe.transform(data_raw[columns_to_encode])
not_encoded_columns = data_raw[columns_not_to_encode]

preprocessed_raw_data = np.concatenate([encoded_columns, not_encoded_columns], axis=1)

X_train, X_test, y_train, y_test = train_test_split(preprocessed_raw_data, Y_values, test_size=.3, random_state=42)
#
