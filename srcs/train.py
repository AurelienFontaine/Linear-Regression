import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

def normalize_data(data):
    """Normalize data using min-max scaling"""
    return (data - data.min()) / (data.max() - data.min())

def denormalize_price(normalized_price, price_min, price_max):
    """Denormalize price back to original scale"""
    return normalized_price * (price_max - price_min) + price_min

def calculate_cost(predictions, actual):
    """Calculate Mean Squared Error"""
    return np.mean((predictions - actual) ** 2)

def train_model(filename, learning_rate=0.01, iterations=1000):
    # Read the dataset
    data = pd.read_csv(filename)
    mileage = data['km'].values  # Using correct column name from CSV
    price = data['price'].values

    # Store original data for denormalization
    mileage_min, mileage_max = mileage.min(), mileage.max()
    price_min, price_max = price.min(), price.max()

    # Normalize the data
    mileage_norm = normalize_data(mileage)
    price_norm = normalize_data(price)

    m = len(mileage_norm)  # Number of training examples
    theta0 = 0  # Initial theta0
    theta1 = 0  # Initial theta1

    # Store cost history for visualization
    cost_history = []

    print("Starting training...")
    print(f"Dataset size: {m} examples")
    print(f"Learning rate: {learning_rate}")
    print(f"Iterations: {iterations}")

    # Gradient Descent
    for i in range(iterations):
        predictions = theta0 + theta1 * mileage_norm
        errors = predictions - price_norm

        # Compute temporary updates
        tmp_theta0 = theta0 - (learning_rate * errors.sum() / m)
        tmp_theta1 = theta1 - (learning_rate * (errors * mileage_norm).sum() / m)

        # Update theta0 and theta1 simultaneously
        theta0, theta1 = tmp_theta0, tmp_theta1

        # Calculate and store cost
        cost = calculate_cost(predictions, price_norm)
        cost_history.append(cost)

        # Print progress every 100 iterations
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}: Cost = {cost:.6f}")

    # Calculate final metrics
    final_predictions = theta0 + theta1 * mileage_norm
    final_cost = calculate_cost(final_predictions, price_norm)
    
    # Calculate R² score
    ss_res = np.sum((price_norm - final_predictions) ** 2)
    ss_tot = np.sum((price_norm - np.mean(price_norm)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)

    print(f"\nTraining completed!")
    print(f"Final cost: {final_cost:.6f}")
    print(f"R² score: {r2_score:.4f}")
    print(f"Normalized parameters: theta0 = {theta0:.4f}, theta1 = {theta1:.4f}")

    # Save the model parameters and normalization data
    model_data = {
        'theta0': theta0,
        'theta1': theta1,
        'mileage_min': mileage_min,
        'mileage_max': mileage_max,
        'price_min': price_min,
        'price_max': price_max
    }
    
    with open('model_parameters.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    # Plotting (Bonus part)
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Data distribution and regression line
    plt.subplot(1, 3, 1)
    plt.scatter(mileage, price, alpha=0.6, color='blue', label='Data points')
    
    # Create regression line for visualization
    mileage_range = np.linspace(mileage.min(), mileage.max(), 100)
    mileage_range_norm = normalize_data(mileage_range)
    price_pred_norm = theta0 + theta1 * mileage_range_norm
    price_pred = denormalize_price(price_pred_norm, price_min, price_max)
    
    plt.plot(mileage_range, price_pred, color='red', linewidth=2, label='Regression line')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price ($)')
    plt.title('Linear Regression: Price vs Mileage')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Cost function over iterations
    plt.subplot(1, 3, 2)
    plt.plot(cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost (MSE)')
    plt.title('Cost Function Convergence')
    plt.grid(True, alpha=0.3)

    # Plot 3: Residuals plot
    plt.subplot(1, 3, 3)
    final_predictions_denorm = denormalize_price(final_predictions, price_min, price_max)
    residuals = price - final_predictions_denorm
    plt.scatter(final_predictions_denorm, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Price ($)')
    plt.ylabel('Residuals ($)')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    return model_data

if __name__ == "__main__":
    dataset_file = "data.csv"
    train_model(dataset_file)
