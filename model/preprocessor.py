import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle

# Load your dataset
data_path = r'C:\Users\suman\Desktop\Mini Project\dataset\synthetic_desalination_data.csv'  # Replace with the correct path
data = pd.read_csv(data_path)

# Define numerical and categorical columns
numerical_features = ['Plant Capacity (m³/day)', 'Project Award Year', 'Raw Water Salinity (mg/L)']
categorical_features = ['Plant Location', 'Plant Type', 'Project Financing Type']

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Fit the preprocessor on the dataset
preprocessor.fit(data)

# Save the preprocessor
preprocessor_path = r'C:\Users\suman\Desktop\Mini Project\model\preprocessor.pkl'
with open(preprocessor_path, 'wb') as f:
    pickle.dump(preprocessor, f)

print(f"Preprocessor saved at {preprocessor_path}")
