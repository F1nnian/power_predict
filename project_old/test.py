import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('powerpredict.csv')

# One-hot encoding for categorical features
data = pd.get_dummies(data, columns=[
    'Bedrock_weather_main', 'Bedrock_weather_description', 
    'Gotham City_weather_main', 'Gotham City_weather_description', 
    'New New York_weather_description', 'New New York_weather_main', 
    'Springfield_weather_description', 'Springfield_weather_main', 
    'Paperopoli_weather_main', 'Paperopoli_weather_description'
], drop_first=True)

# Prepare features and target
X = data.drop('power_consumption', axis=1).values
y = data['power_consumption'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% of the variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the Neural Network
class PowerConsumptionNN(nn.Module):
    def __init__(self, input_dim):
        super(PowerConsumptionNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

# Hyperparameters
input_dim = X_train_pca.shape[1]
batch_size = 32
learning_rate = 0.001
num_epochs = 100

# Data loaders
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Model, loss function, optimizer
model = PowerConsumptionNN(input_dim=input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_nn = model(X_test_tensor).numpy()
mse_nn = mean_squared_error(y_test, y_pred_nn)
print(f'Neural Network MSE: {mse_nn}')

# Visualization
plt.figure(figsize=(14, 6))

# Plot true vs predicted power consumption for Neural Network
plt.scatter(y_test, y_pred_nn, alpha=0.3, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Neural Network')

plt.show()
