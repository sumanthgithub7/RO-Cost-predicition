import pickle
import pandas as pd

# Load the preprocessor
preprocessor_path = r'C:\Users\suman\Desktop\Mini Project\model\preprocessor.pkl'
with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

# Sample input for testing
input_data = pd.DataFrame([{
    'Plant Location': 'USA',
    'Plant Capacity (m³/day)': 5000,
    'Project Award Year': 2020,
    'Raw Water Salinity (mg/L)': 20000,
    'Plant Type': 'Brackish water',
    'Project Financing Type': 'Public'
}])

# Transform the input data
preprocessed_data = preprocessor.transform(input_data)
print(preprocessed_data.shape)  # Check the number of features
