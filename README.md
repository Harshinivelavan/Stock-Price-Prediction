# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset


Predict future stock prices using an RNN model based on historical closing prices from trainset.csv and testset.csv, with data normalized using MinMaxScaler.


## Design Steps

### Step 1:

Import necessary libraries.

### Step 2:

Load and preprocess the data.

### Step 3:

Create input-output sequences.

### Step 4:

Convert data to PyTorch tensors.

### Step 5:

Define the RNN model.

### Step 6:

Train the model using the training data.

### Step 7:

Evaluate the model and plot predictions.



## Program


#### Name:Harshini V

#### Register Number:212224040109

```Python 
# Define RNN Model
class RNNModel(nn.Module):
    class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)

        # Take the last time step output
        out = out[:, -1, :]

        # Pass through fully connected layer
        out = self.fc(out)

        return out





model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
train_losses = []

model.train()

for epoch in range(num_epochs):
    epoch_loss = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')






```

## Output


<img width="486" height="457" alt="image" src="https://github.com/user-attachments/assets/dc9f994b-9f9b-4a90-a9cd-5ca3ecff0dca" />



### True Stock Price, Predicted Stock Price vs time



<img width="845" height="645" alt="image" src="https://github.com/user-attachments/assets/a9e30158-2731-43de-bb7f-ad4aa88f2ff8" />


### Predictions 




<img width="1193" height="782" alt="image" src="https://github.com/user-attachments/assets/72adb0b2-1b39-4b6b-bbfc-fa11a4f4f170" />


## Result

Thus, a Recurrent Neural Network model for stock price prediction has successfully been devoloped.
