import torch
from model import NeuralNet  # Import your model class

# Load the model from the .pth file
model_data = torch.load("processed_results1.pth")
input_size = model_data['input_size']
hidden_size = model_data['hidden_size']
output_size = model_data['output_size']

# Initialize your model
model = NeuralNet(input_size, hidden_size, output_size)

try:
    # Load the model's state dictionary
    model.load_state_dict(model_data['model_state'])
    print("Model loaded successfully!")
except RuntimeError as e:
    print("Error loading the model state dictionary:", e)
