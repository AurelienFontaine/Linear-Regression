import pickle
import numpy as np

def normalize_mileage(mileage, mileage_min, mileage_max):
    """Normalize mileage using the same scaling as training"""
    return (mileage - mileage_min) / (mileage_max - mileage_min)

def denormalize_price(normalized_price, price_min, price_max):
    """Denormalize price back to original scale"""
    return normalized_price * (price_max - price_min) + price_min

def predict_price():
    try:
        # Load the trained parameters and normalization data
        with open('model_parameters.pkl', 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        print("Model parameters not found. Please train the model first.")
        return

    # Extract model parameters and normalization data
    theta0 = model_data['theta0']
    theta1 = model_data['theta1']
    mileage_min = model_data['mileage_min']
    mileage_max = model_data['mileage_max']
    price_min = model_data['price_min']
    price_max = model_data['price_max']

    print("Model loaded successfully!")
    print(f"Model parameters: theta0 = {theta0:.4f}, theta1 = {theta1:.4f}")
    print(f"Data range: Mileage [{mileage_min:.0f} - {mileage_max:.0f}] km, Price [${price_min:.0f} - ${price_max:.0f}]")

    while True:
        try:
            # Prompt the user for mileage
            mileage_input = input("\nEnter the mileage of the car (or 'quit' to exit): ")
            
            if mileage_input.lower() == 'quit':
                print("Goodbye!")
                break
                
            mileage = float(mileage_input)
            
            # Check if mileage is within reasonable range
            if mileage < 0:
                print("Warning: Mileage cannot be negative!")
                continue
                
            if mileage > mileage_max * 1.5:
                print(f"Warning: Mileage {mileage:.0f} km is much higher than training data max ({mileage_max:.0f} km)")
                print("Prediction may be less accurate.")
            
            # Normalize the input mileage
            mileage_norm = normalize_mileage(mileage, mileage_min, mileage_max)
            
            # Calculate the estimated price (normalized)
            estimated_price_norm = theta0 + (theta1 * mileage_norm)
            
            # Denormalize the price
            estimated_price = denormalize_price(estimated_price_norm, price_min, price_max)
            
            # Ensure price is not negative
            estimated_price = max(0, estimated_price)
            
            print(f"Estimated price for mileage {mileage:.0f} km is: ${estimated_price:.2f}")
            
                
        except ValueError:
            print("Please enter a valid number for mileage.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    predict_price()
