import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data
mpg = [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8,  
       19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4,
       33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8,  
      19.7, 15.0, 21.4]
hp = [110, 110, 93, 110, 175, 105, 245, 62, 95, 123, 123, 180,  
      180, 180, 205, 215, 230, 66, 52, 65, 97, 150, 150, 245,
      175, 66, 91, 113, 264, 175, 335, 109]
weight = [2700, 2750, 2500, 2900, 3500, 2800, 4000, 2300, 2800, 3000, 
          2900, 3300, 3100, 3500, 3600, 3800, 3500, 2200, 2400, 2500, 
          2700, 3000, 2800, 3200, 2900, 2800, 3100, 3400, 3300, 3600, 
          3000, 2700, 3100]

# Convert the data into a DataFrame for better management
data = pd.DataFrame({
    'mpg': mpg,
    'hp': hp,
    'weight': weight
})

# Outlier Detection and Removal (Z-Score)
z_scores = np.abs(stats.zscore(data))
data_cleaned = data[(z_scores < 3).all(axis=1)]  # Keep only rows without outliers

# Pearson correlation
correlation, p_value = stats.pearsonr(data_cleaned['mpg'], data_cleaned['hp'])
print(f'Pearson correlation: {correlation}')
print(f'P-value: {p_value}')

# Linear Regression
X = data_cleaned[['hp', 'weight']]  # Features
y = data_cleaned['mpg']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r_squared = model.score(X_test, y_test)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r_squared}')

# Visualize the relationship
sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))

# Scatter plot with regression line
sns.regplot(x='hp', y='mpg', data=data_cleaned, line_kws={'color': 'red'})
plt.xlabel('Horsepower (hp)')
plt.ylabel('Mileage (mpg)')
plt.title('Scatter plot of Mileage vs Horsepower with Regression Line')
plt.show()

# Histogram of mpg and hp
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(data_cleaned['mpg'], bins=10, kde=True, color='blue', ax=axes[0]).set_title('Histogram of MPG')
sns.histplot(data_cleaned['hp'], bins=10, kde=True, color='green', ax=axes[1]).set_title('Histogram of Horsepower')
plt.show()

# Conclusion: Correlation and predictions
if p_value < 0.05:
    print("The relationship between mpg and hp is statistically significant.")
else:
    print("The relationship between mpg and hp is not statistically significant.")

