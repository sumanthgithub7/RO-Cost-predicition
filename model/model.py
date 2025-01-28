import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

# Load the dataset
data = pd.read_csv(r'C:\Users\suman\Desktop\Mini Project\dataset\synthetic_desalination_data.csv')
print(data.head())

# Specify input and output columns
input_columns = ['Plant Location', 'Plant Capacity (m³/day)', 'Project Award Year',
                 'Raw Water Salinity (mg/L)', 'Plant Type', 'Project Financing Type']
output_column = 'Capital Cost (USD)'

# Separate features and target
X = data[input_columns]
y = data[output_column]

# Preprocessing: Handle categorical and numeric data
categorical_columns = ['Plant Location', 'Plant Type', 'Project Financing Type']
numeric_columns = ['Plant Capacity (m³/day)', 'Project Award Year', 'Raw Water Salinity (mg/L)']

# Numeric transformer: Scale numerical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Categorical transformer: OneHotEncode categorical features and handle unknown categories
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both transformers into a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Preprocess the features (X)
X_processed = preprocessor.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create DataLoader for mini-batch training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define Multi-Layer Feedforward Neural Network with dropout and L2 regularization
class FeedforwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(p=0.5)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(p=0.5)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 1)  # Output layer

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.dropout3(self.relu3(self.fc3(x)))
        x = self.fc4(x)  # No activation on the output layer for regression
        return x

# Initialize model, loss function, and optimizer
input_size = X_train_tensor.shape[1]  # Automatically determine input size
model = FeedforwardNN(input_size)
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)  # L2 regularization (weight decay)

# Early stopping parameters
best_loss = float('inf')
patience = 50
counter = 0

# Training loop with mini-batches and early stopping
epochs = 1000
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

    # Evaluate validation loss
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_test_tensor)
        val_loss = criterion(val_predictions, y_test_tensor)

    # Early stopping logic
    if val_loss.item() < best_loss:
        best_loss = val_loss.item()
        counter = 0
        # Save the best model state
        torch.save(model.state_dict(), r'C:\Users\suman\Desktop\Mini Project\model\model.pth')
    else:
        counter += 1

    if counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Final evaluation metrics
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    print(f'Final Test Loss (MSE): {test_loss.item():.4f}')

    # Convert predictions and targets to numpy for metric calculation
    y_test_np = y_test_tensor.numpy()
    test_predictions_np = test_predictions.numpy()

    r2 = r2_score(y_test_np, test_predictions_np)
    mae = mean_absolute_error(y_test_np, test_predictions_np)

    print(f"R^2 Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.2f}")

# Save the trained model (last epoch)
torch.save(model.state_dict(),r'C:\Users\suman\Desktop\Mini Project\model.pth')
print("Model saved to model.pth and best model saved to model_best.pth")
