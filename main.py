import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Define a custom dataset
class SignalDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(
            label, dtype=torch.long
        )


# Load data and assign labels
def load_data():
    files_labels = {
        "Cycler": [
            "./PulseTrainData/Processed/Cycler1.csv",
            "./PulseTrainData/Processed/Cycler2.csv",
        ],
        "Jitter": [
            "./PulseTrainData/Processed/Jitter1.csv",
            "./PulseTrainData/Processed/Jitter2.csv",
            "./PulseTrainData/Processed/Jitter3.csv",
        ],
        "Stable": [
            "./PulseTrainData/Processed/Stable1.csv",
            "./PulseTrainData/Processed/Stable2.csv",
            "./PulseTrainData/Processed/Stable3.csv",
        ],
        "Stagger": [
            "./PulseTrainData/Processed/Stagger1.csv",
            "./PulseTrainData/Processed/Stagger2.csv",
        ],
    }

    data = []
    labels = []
    label_mapping = {"Cycler": 0, "Jitter": 1, "Stable": 2, "Stagger": 3}

    for label, files in files_labels.items():
        for file in files:
            df = pd.read_csv(file)
            data.append(df.values)
            labels.append(np.full(df.shape[0], label_mapping[label]))

    data = np.vstack(data)
    labels = np.concatenate(labels)

    return data, labels


# Define the neural network
class SignalClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SignalClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}"
        )


# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")


# Function to predict signal type for a new CSV file
def predict_signal(model, scaler, file_path):
    df = pd.read_csv(file_path)
    # Aggregate features by computing the mean across rows
    aggregated_data = df.mean(axis=0).values.reshape(1, -1)
    data = scaler.transform(aggregated_data)
    inputs = torch.tensor(data, dtype=torch.float32)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    label_mapping = {0: "Cycler", 1: "Jitter", 2: "Stable", 3: "Stagger"}
    return label_mapping[predicted.item()]


def data_preprocessing():
    # Load CSV
    input_file = "./PulseTrainData/Interleaved/Interleaved8.csv"
    df = pd.read_csv(input_file)

    # Set column key
    angle_key = "angle"

    # Store DataFrames by angle
    grouped_dfs = {angle: data for angle, data in df.groupby(angle_key)}

    for angle, group_df in grouped_dfs.items():
        output_file = f"./PulseTrainData/Interleaved/Interleaved8_{angle}.csv"
        group_df.to_csv(output_file, index=False)
        print(f"Saved {output_file}")

    print("Data preprocessing complete.")


# Main function
def main():
    # Only run if new data exists in Interleaved.
    # data_preprocessing()

    # Load and preprocess data
    data, labels = load_data()

    # Normalize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    # Create datasets and dataloaders
    train_dataset = SignalDataset(X_train, y_train)
    test_dataset = SignalDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Determine the input size from data
    input_size = X_train.shape[1]
    num_classes = 4

    # Initialize the model
    model = SignalClassifier(input_size, num_classes)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=20)

    # Evaluate the model
    evaluate_model(model, test_loader)

    # Save the trained model
    torch.save(model.state_dict(), "signal_classifier.pth")

    directory_path = "./PulseTrainData/Interleaved/Processed"

    # Glob pattern to match CSV files
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    # Iterate over and make predictions for each CSV file
    for csv_file in csv_files:
        predicted_label = predict_signal(model, scaler, csv_file)
        print(f"Predicted label for {csv_file}: {predicted_label}")

    print("Prediction complete.")


if __name__ == "__main__":
    main()
