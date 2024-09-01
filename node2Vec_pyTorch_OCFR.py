import torch
import torch.nn as nn
import torch.optim as optim

# Assume 'embedding' is the node embedding from Node2Vec as a numpy array
# and 'labels' is a corresponding array of labels for supervised learning
embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)  # Assuming classification labels

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Instantiate the model
model = SimpleNN(input_size=64, hidden_size=32, output_size=10)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

# Training loop
num_epochs = 100  # Number of times to iterate over the dataset

for epoch in range(num_epochs):
    # Forward pass: compute predicted output by passing embeddings through the model
    outputs = model(embedding_tensor)
    
    # Compute loss
    loss = criterion(outputs, labels_tensor)
    
    # Backward pass: compute gradient of the loss with respect to model parameters
    optimizer.zero_grad()  # Clear existing gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update model parameters
    
    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
