import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Step 1: Load the dataset
gold_data = pd.read_csv('gld_price_data (1).csv')

# Step 2: Explore the dataset
print("First 5 rows of the dataset:")
print(gold_data.head())
print("\nDataset Info:")
print(gold_data.info())

# Step 3: Check for missing values
print("\nMissing values in the dataset:")
print(gold_data.isnull().sum())

# Step 4: Correlation analysis
correlation = gold_data.select_dtypes(include='number').corr()
plt.figure(figsize=(8, 8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.title("Correlation Heatmap")
plt.show()

# Print correlation with the target variable 'GLD'
print("\nCorrelation with GLD:")
print(correlation['GLD'])

# Step 5: Data preparation
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

# Step 6: Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Step 7: Model training
regressor = RandomForestRegressor(n_estimators=100, random_state=2)
regressor.fit(X_train, Y_train)

# Step 8: Model prediction
test_data_prediction = regressor.predict(X_test)

# Step 9: Evaluate the model
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error: ", error_score)

# Step 10: Visualize the results
Y_test = list(Y_test)
plt.figure(figsize=(10, 6))
plt.plot(Y_test, color='blue', label='Actual Value')
plt.plot(test_data_prediction, color='red', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()

