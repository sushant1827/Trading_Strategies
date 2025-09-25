import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from keras import Sequential
from keras.src.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

threshold = 0.5

# Load the data
data = pd.read_csv('SomeStrategyD.csv')
data.drop(columns=['Unnamed: 0'], inplace=True)
# print(data.head())

# ---------------------------------------------------------------------

from sklearn.preprocessing import MinMaxScaler

exclude_columns = ['Date', 'UT_Buy', 'UT_Sell', 'sqzOn', 'sqzOff', 'noSqz', 'green', 'lime', 'red', 'maroon', 'black', 'blue', 'gray']

# Separate columns to scale and columns to exclude
columns_to_scale = [col for col in data.columns if col not in exclude_columns]
columns_to_exclude = [col for col in exclude_columns]

# Create DataFrames for scaling and non-scaling columns
data_to_scale = data[columns_to_scale]
data_to_exclude = data[columns_to_exclude]

# Initialize the scaler (Min-Max Scaling in this case)
scaler = MinMaxScaler()

# Apply scaling only to the selected columns
data_scaled = pd.DataFrame(scaler.fit_transform(data_to_scale), columns=columns_to_scale)

# Concatenate the scaled columns with the excluded columns
data = pd.concat([data_scaled, data_to_exclude.reset_index(drop=True)], axis=1)

# ---------------------------------------------------------------------

data['Pred'] = data['Open'] < data['Close']
data['Pred'] = data['Pred'].astype(int).shift(-1)
data.drop(data.tail(1).index,inplace=True)

# data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

data.to_csv('ML_EnsembleD.csv')

# Drop Date column
# data = data.drop(columns=['Date'])

# data.set_index('Date', inplace=True)

# Separate features and target
# X = data.drop(columns=['Pred'])
# y = data['Pred'].shift(-1)

def create_lagged_features(data, lag=5):
    features = []
    for i in range(lag, len(data)):
        # Select the previous 'lag' rows and flatten them into a single feature vector
        # feature = data.iloc[i-lag:i].drop(columns=['Date', 'Pred']).values.flatten()
        feature = data.iloc[i-lag:i].values.flatten()
        features.append(feature)
    return np.array(features)

lag = 7  # Use the last 5 rows for prediction
X = create_lagged_features(data, lag=lag)
y = data['Pred'].iloc[lag:].values  # Align y with the lagged X

# X.drop(data.tail(1).index,inplace=True)
# y.drop(data.tail(1).index,inplace=True)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Reshape data for LSTM
X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Create and train LSTM model
lstm_model = create_lstm_model((1, X_train.shape[1]))
lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, verbose=1)

# Predict with LSTM model
lstm_predictions = lstm_model.predict(X_test_lstm)
lstm_predictions = [1 if x > threshold else 0 for x in lstm_predictions]

# Train RandomForest, XGBoost, LightGBM models
rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(random_state=42)
lgbm_model = LGBMClassifier(random_state=42)

rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
lgbm_model.fit(X_train, y_train)

# Predict with RandomForest, XGBoost, LightGBM models
rf_predictions = rf_model.predict_proba(X_test)[:, 1]
xgb_predictions = xgb_model.predict_proba(X_test)[:, 1]
lgbm_predictions = lgbm_model.predict_proba(X_test)[:, 1]

# Apply threshold of 0.8
rf_predictions = [1 if x > threshold else 0 for x in rf_predictions]
xgb_predictions = [1 if x > threshold else 0 for x in xgb_predictions]
lgbm_predictions = [1 if x > threshold else 0 for x in lgbm_predictions]

# Ensemble predictions
ensemble_predictions = np.mean([lstm_predictions, rf_predictions, xgb_predictions, lgbm_predictions], axis=0)
ensemble_predictions = [1 if x > threshold else 0 for x in ensemble_predictions]

# Evaluate the ensemble model
accuracy = accuracy_score(y_test, ensemble_predictions)
precision = precision_score(y_test, ensemble_predictions)
recall = recall_score(y_test, ensemble_predictions)
f1 = f1_score(y_test, ensemble_predictions)
roc_auc = roc_auc_score(y_test, ensemble_predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC-ROC: {roc_auc}")

# ------------------------------------------------------

# ensemble_predictions_df = pd.DataFrame(ensemble_predictions, columns=['Ensemble_Predictions'])
# print(len(ensemble_predictions_df))

# # Convert y_test to a DataFrame
# y_test_df = pd.DataFrame(y_test, columns=['True_Labels'])


# # Reset the index of X_test (and optionally y_test, if necessary)
# X_test_reset = X_test.reset_index(drop=True)
# y_test_reset = X_test.reset_index(drop=True)

# X_test_reset = scaler.inverse_transform(X_test)


# # Combine X_test, y_test, and ensemble_predictions
# combined_df = pd.concat([X_test_reset, y_test_df, ensemble_predictions_df], axis=1)

# # Export to a CSV file
# combined_df.to_csv('combined_X_test_y_test_ensemble_predictions.csv', index=False)


# # Convert y_test to a DataFrame
# y_test_df = pd.DataFrame(y_test, columns=['True_Labels']).reset_index(drop=True)
# print(len(y_test_df))

# # Reset the index of X_test
# X_test_df = pd.DataFrame(X_test, columns=X.columns)
# X_test_reset = X_test_df.reset_index(drop=True)

# # # Inverse transform X_test to restore original values
# # X_test_original = pd.DataFrame(scaler.inverse_transform(X_test_reset), columns=X_test.columns)
# # print(len(X_test_original))

# # # Combine X_test, y_test, and ensemble_predictions
# # combined_df = pd.concat([X_test_original, y_test_df, ensemble_predictions_df], axis=1)

# # # Export to a CSV file
# # combined_df.to_csv('combined_X_test_y_test_ensemble_predictions.csv', index=False)


# # Correctly identifying the columns that were originally scaled
# scaled_columns = X_test.columns.intersection(columns_to_scale)
# unscaled_columns = X_test.columns.difference(columns_to_scale)

# # Separate scaled and unscaled parts of X_test_reset
# X_test_scaled_part = X_test_reset[scaled_columns]
# X_test_unscaled_part = X_test_reset[unscaled_columns]

# # Inverse transform the scaled part
# X_test_original_scaled = pd.DataFrame(scaler.inverse_transform(X_test_scaled_part), columns=scaled_columns)

# # Combine the inverse-transformed part with the unscaled part
# X_test_original = pd.concat([X_test_original_scaled, X_test_unscaled_part], axis=1)
# print(len(X_test_original))

# # Combine X_test, y_test, and ensemble_predictions
# combined_df = pd.concat([X_test_original, y_test_df, ensemble_predictions_df], axis=1)

# # Export to a CSV file
# combined_df.to_csv('combined_X_test_y_test_ensemble_predictions.csv', index=False)


