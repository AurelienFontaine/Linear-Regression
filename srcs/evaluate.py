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

def calculate_metrics(y_true, y_pred):
    """Calculate various evaluation metrics"""
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # R² Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2_score
    }

def evaluate_model(dataset_file='data.csv'):
    """Evaluate the trained model on the dataset"""
    try:
        # Load the trained model
        with open('model_parameters.pkl', 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        print("Model parameters not found. Please train the model first.")
        return

    # Load the dataset
    data = pd.read_csv(dataset_file)
    mileage = data['km'].values
    price = data['price'].values

    # Extract model parameters and normalization data
    theta0 = model_data['theta0']
    theta1 = model_data['theta1']
    mileage_min = model_data['mileage_min']
    mileage_max = model_data['mileage_max']
    price_min = model_data['price_min']
    price_max = model_data['price_max']

    # Normalize the data
    mileage_norm = normalize_data(mileage)
    price_norm = normalize_data(price)

    # Make predictions
    predictions_norm = theta0 + theta1 * mileage_norm
    predictions = denormalize_price(predictions_norm, price_min, price_max)

    # Calculate metrics
    metrics = calculate_metrics(price, predictions)

    # Print results
    # print("=" * 50)
    # print("MODEL EVALUATION RESULTS")
    # print("=" * 50)
    print(f"Dataset size: {len(price)} examples")
    print(f"Model parameters: theta0 = {theta0:.4f}, theta1 = {theta1:.4f}")
    print()
    print("PERFORMANCE METRICS:")
    print("-" * 30)
    print(f"Mean Squared Error (MSE):     ${metrics['MSE']:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${metrics['RMSE']:.2f}")
    print(f"Mean Absolute Error (MAE):     ${metrics['MAE']:.2f}")
    print(f"Mean Absolute Percentage Error: {metrics['MAPE']:.2f}%")
    print(f"R² Score:                      {metrics['R2']:.4f}")
    print()
    
    # Interpretation
    print("INTERPRETATION:")
    print("-" * 30)
    if metrics['R2'] > 0.8:
        print("✓ Excellent model fit (R² > 0.8)")
    elif metrics['R2'] > 0.6:
        print("✓ Good model fit (R² > 0.6)")
    elif metrics['R2'] > 0.4:
        print("⚠ Moderate model fit (R² > 0.4)")
    else:
        print("⚠ Poor model fit (R² < 0.4)")
    
    print(f"• On average, predictions are off by ${metrics['MAE']:.2f}")
    print(f"• The model explains {metrics['R2']*100:.1f}% of the variance in car prices")
    
    if metrics['MAPE'] < 10:
        print("✓ Very accurate predictions (MAPE < 10%)")
    elif metrics['MAPE'] < 20:
        print("✓ Good accuracy (MAPE < 20%)")
    else:
        print("⚠ Moderate accuracy (MAPE > 20%)")

    # Create evaluation plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(2, 3, 1)
    plt.scatter(price, predictions, alpha=0.6, color='blue')
    plt.plot([price.min(), price.max()], [price.min(), price.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Actual vs Predicted Prices')
    plt.grid(True, alpha=0.3)
    
    # Add R² to the plot
    plt.text(0.05, 0.95, f'R² = {metrics["R2"]:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Residuals
    plt.subplot(2, 3, 2)
    residuals = price - predictions
    plt.scatter(predictions, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Price ($)')
    plt.ylabel('Residuals ($)')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)

    # Plot 3: Residuals histogram
    plt.subplot(2, 3, 3)
    plt.hist(residuals, bins=20, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Residuals ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)

    # Plot 4: Error by mileage
    plt.subplot(2, 3, 4)
    absolute_errors = np.abs(residuals)
    plt.scatter(mileage, absolute_errors, alpha=0.6, color='orange')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Absolute Error ($)')
    plt.title('Prediction Error vs Mileage')
    plt.grid(True, alpha=0.3)

    # Plot 5: Price range analysis
    plt.subplot(2, 3, 5)
    price_ranges = ['Low', 'Medium', 'High']
    low_mask = price < np.percentile(price, 33)
    high_mask = price > np.percentile(price, 67)
    medium_mask = ~(low_mask | high_mask)
    
    ranges_mae = [
        np.mean(np.abs(residuals[low_mask])),
        np.mean(np.abs(residuals[medium_mask])),
        np.mean(np.abs(residuals[high_mask]))
    ]
    
    plt.bar(price_ranges, ranges_mae, color=['lightblue', 'lightgreen', 'lightcoral'])
    plt.ylabel('Mean Absolute Error ($)')
    plt.title('Error by Price Range')
    plt.grid(True, alpha=0.3)

    # Plot 6: Model performance summary
    plt.subplot(2, 3, 6)
    metrics_names = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²']
    metrics_values = [metrics['MSE'], metrics['RMSE'], metrics['MAE'], metrics['MAPE'], metrics['R2']]
    
    # Normalize values for visualization (except R²)
    normalized_values = []
    for i, (name, value) in enumerate(zip(metrics_names, metrics_values)):
        if name == 'R²':
            normalized_values.append(value)
        else:
            # Normalize to 0-1 scale for better visualization
            max_val = max(metrics_values[:-1])  # Exclude R²
            normalized_values.append(value / max_val)
    
    bars = plt.bar(metrics_names, normalized_values, color=['red', 'orange', 'yellow', 'lightblue', 'green'])
    plt.ylabel('Normalized Score')
    plt.title('Model Performance Summary')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add actual values as text
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

    return metrics

if __name__ == "__main__":
    evaluate_model()
