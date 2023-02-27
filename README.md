# Heart-Disease-Prediction

The following code is written in Python and uses various machine learning libraries to train a logistic regression model to predict whether a person has a heart disease or not. Here is a detailed documentation of the code:

# Importing Libraries

numpy: A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

pandas: A fast, powerful, flexible, and easy-to-use open-source data analysis and manipulation tool, built on top of the Python programming language.

train_test_split: A function in the sklearn.model_selection module that splits the dataset into random train and test subsets.
 
LogisticRegression: A class in the sklearn.linear_model module that implements logistic regression for classification tasks.

accuracy_score: A function in the sklearn.metrics module that calculates the accuracy of the classification model.

# Loading Dataset

 heart_data: Reads the CSV file heart_disease_data.csv and stores the data in a pandas DataFrame.
 
 shape: Prints the dimensions of the DataFrame.
 
 info: Prints a summary of the DataFrame, including the number of non-null values in each column.
 
 describe: Prints the descriptive statistics of the DataFrame.
 
 
 value_counts: Prints the count of each unique value in the target column.
 
 
 # Preparing Data

X: Stores all the features (columns except the target column) of the DataFrame.

Y: Stores the target column of the DataFrame.

drop: Drops the target column from the DataFrame.

axis: Specifies that the target column should be dropped from the columns (not the rows).

train_test_split: Splits the dataset into random train and test subsets, with a 80/20 split ratio and stratification based on the target column.

model: Initializes a logistic regression model.


# Training the Model

fit: Fits the logistic regression model to the training data.

X_train_prediction: Predicts the target values of the training data using the trained model.

training_data_accuracy: Calculates the accuracy of the model on the training data.

X_test_prediction: Predicts the target values of the test data using the trained model.

test_data_accuracy: Calculates the accuracy of the model on the test data.

# Making Predictions

input_data: Stores the input data to be predicted.

asarray: Converts the input data to a NumPy array.

reshape: Reshapes the NumPy array to have a shape of (1, -1) as the model expects a 2D array as input.

prediction: Predicts the target value for the input data using the trained model.

if-else statement: Prints whether the person has a heart disease or not based on the predicted value.
 
