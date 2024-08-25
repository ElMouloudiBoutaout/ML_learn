import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Example data
hours_studied = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Reshaping for sklearn
test_scores = np.array([50, 55, 65, 70, 75])

# Create and train the model
model = LinearRegression()
model.fit(hours_studied, test_scores)

# Predict test scores for the given hours
predicted_scores = model.predict(hours_studied)

# Print the model parameters
print(f"Slope (m): {model.coef_[0]}")
print(f"Intercept (b): {model.intercept_}")

# Predicting the score for 6 hours of study
hours_for_prediction = np.array([[6],[2]])
predicted_score_for_6_hours = model.predict(hours_for_prediction)
print(f"Predicted score for 6 hours of study: {predicted_score_for_6_hours[0]}")

# Plotting the data and the regression line
plt.scatter(hours_studied, test_scores, color='blue', label='Actual Scores')
plt.plot(hours_studied, predicted_scores, color='red', label='Regression Line')
plt.scatter(hours_for_prediction, predicted_score_for_6_hours, color='green', label='Prediction for 6 Hours')
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.legend()
plt.show()
