import numpy as np

# Initialize random seed for reproducibility
np.random.seed(42)

# Step 1: Define the input data (x) and corresponding target values (y_target)
x_data = np.array([1.0, 2.0, 3.0, 4.0])
y_target = np.array([2.0, 4.0, 6.0, 8.0])

# Step 2: Initialize weight (m) and bias (b)
m = np.random.rand()  # Randomly initialize weight
b = np.random.rand()  # Randomly initialize bias

print(f"Initial weight (m): {m}")
print(f"Initial bias (b): {b}")

# Step 3: Define the model (prediction function) and the loss function
def predict(x):
    return m * x + b

def compute_loss(y_pred, y_target):
    return np.mean((y_pred - y_target) ** 2)

# Step 4: Training the model using gradient descent
learning_rate = 0.01  # Learning rate
num_iterations = 10000  # Number of iterations

for i in range(num_iterations):
    # Make predictions
    y_pred = predict(x_data)
    
    # Calculate the loss
    loss = compute_loss(y_pred, y_target)
    
    # Calculate the gradients
    gradient_m = np.mean(2 * (y_pred - y_target) * x_data)
    gradient_b = np.mean(2 * (y_pred - y_target))
    
    # Update weight and bias
    global m, b
    m -= learning_rate * gradient_m
    b -= learning_rate * gradient_b
    
    # Print the loss every 100 iterations
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss}, m = {m}, b = {b}")

# Step 5: Final predictions after training
final_predictions = predict(x_data)
final_loss = compute_loss(final_predictions, y_target)

print("\nFinal model:")
print(f"Weight (m): {m}")
print(f"Bias (b): {b}")
print(f"Final Loss: {final_loss}")
print(f"Predictions: {final_predictions}")
