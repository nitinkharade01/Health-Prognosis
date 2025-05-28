import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
from spell_checker import MedicalSpellChecker
import nltk

# Download NLTK resources
nltk.download('punkt')

# Initialize spell checker
spell_checker = MedicalSpellChecker()

# Load intents data
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Initialize lists to hold processed data
all_words = []
tags = []
xy = []

# Loop through each sentence in intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # Add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # Correct spelling in pattern
        corrected_pattern = spell_checker.correct_text(pattern)
        # Tokenize each word in the sentence
        words = tokenize(corrected_pattern)
        # Add to the words list
        all_words.extend(words)
        # Add to the xy pair (words, tag)
        xy.append((words, tag))

# Stem and lower each word, ignoring punctuation
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Remove duplicates and sort the words list
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Print out some basic statistics
print(f"{len(xy)} patterns")
print(f"{len(tags)} tags: {tags}")
print(f"{len(all_words)} unique stemmed words: {all_words}")

# Prepare training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    # Convert the sentence into a bag of words
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # Get the corresponding tag as a label (CrossEntropyLoss needs class labels)
    label = tags.index(tag)
    y_train.append(label)

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

# Print the sizes
print(f"Input size: {input_size}, Output size: {output_size}")

# Custom Dataset class for training
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Initialize the dataset and dataloader
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Set device for training (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Final loss
print(f'Final loss: {loss.item():.4f}')

# Save the trained model and related data
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tag": tags,
    "spell_checker": spell_checker  # Save spell checker instance
}

FILE = "chatbot_model.pth"
torch.save(data, FILE)

print(f'Training complete. Model saved to {FILE}')
