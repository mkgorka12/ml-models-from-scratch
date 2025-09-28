import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error

from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from models.linear_regression import LinearRegression as MlFunLinearRegression
from models.logistic_regression import LogisticRegression as MlFunLogisticRegression

import time

print('COMPARE ML FUN LINEAR MODELS TO SCIKIT\n')

# Upload data into DataFrame
filepath = input('Enter the .csv file name:\n> ')
try:
    df = pd.read_csv(filepath).select_dtypes(include='number') # Numeric values only
except Exception as e:
    print(e)
    exit(1)

# Print the head of the uploaded DataFrame
print(f'\n{df.head()}\n')

# Show basic metadata
print(df.describe(), end='\n\n') # Print basic metadata

df_corr = df.corr(numeric_only=True) # Get correlation matrix with numerical only data
print(df_corr, end='\n\n') # Print correlation matrix

# Get the label to train and predict on
print('What label do you want to train?:')
for idx, label in enumerate(df.columns):
    print(f'[{idx}] {label}')
print('> ', end='')

try:
    choosen_label = df.columns[int(input())]
except Exception as e:
    print(e)
    exit(2)

# Get correlated with label features and sort them from the most correlated to the least one
corr_with_label = df_corr[choosen_label].drop(choosen_label).abs().sort_values(ascending=False)

# Get top n most correlated features to the label
print('\nHow many most correlated features do you want to use?:')

most_corr_features_dict = dict()
for i in range(len(corr_with_label)):
    # Append the list of the most correlated features in the descending order to the dictionary
    most_corr_features_dict[i] = corr_with_label.head(len(corr_with_label) - i).to_dict().keys()
    print(f"[{i}]: { list(most_corr_features_dict[i]) }")

try:
    top_n_corr_names = most_corr_features_dict[int(input('> '))]
except Exception as e:
    print(e)
    exit(3)

# Visualise data
pd.plotting.scatter_matrix(df[[*top_n_corr_names, choosen_label]], figsize=(8, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()

# Choose model
valid_models_choices = [(SklearnLinearRegression(), MlFunLinearRegression()),\
                       (SklearnLogisticRegression(), MlFunLogisticRegression())]
print('\nWhat models do you want to use?:')
for idx, valid_models_choice in enumerate(valid_models_choices):
    print(f'[{idx}]: {valid_models_choice}')

try:
    scikit_model, author_model = valid_models_choices[int(input('> '))]
except Exception as e:
    print(e)
    exit(4)

# Prepare data 
x = df[top_n_corr_names].to_numpy()
y = df[choosen_label].to_numpy()

x_len = x.shape[0]
y_len = y.shape[0]

test_to_train_ratio = 0.2 # 80% of the data is a training set, 20% left is a test set

x_train = x[:-int(x_len * test_to_train_ratio)]
x_test = x[int(x_len * test_to_train_ratio):]

y_train = y[:-int(y_len * test_to_train_ratio)]
y_test = y[int(y_len * test_to_train_ratio):]

# Train model
try:
    start = time.time_ns()
    scikit_model = scikit_model.fit(x_train, y_train)
    training_time = (time.time_ns() - start) / 1000000

    print(f"\nScikit model trained in {training_time} ms")

    start = time.time_ns()
    author_model = author_model.fit(x_train, y_train)
    training_time = (time.time_ns() - start) / 1000000

    print(f"ML-Fun model trained in {training_time} ms", end='\n\n')
except Exception as e:
    print(e)
    exit(5)
    
# Compare results
try:
    scikit_model_prediction = scikit_model.predict(x_test)
    author_model_prediction = author_model.predict(x_test) 

    print(f"Scikit model's RMSE: {root_mean_squared_error(y_test, scikit_model_prediction)}")
    print(f"ML-Fun model's RMSE {root_mean_squared_error(y_test, author_model_prediction)}\n")
except Exception as e:
    print(e)
    exit(6)
