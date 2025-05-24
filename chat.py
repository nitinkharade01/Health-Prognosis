import torch
import json
import random
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
from spell_checker import MedicalSpellChecker

def load_model():
    try:
        # Load the model and data
        data = torch.load("chatbot_model.pth")
        
        # Get the saved data
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        all_words = data["all_words"]
        tags = data["tag"]
        spell_checker = data.get("spell_checker", MedicalSpellChecker())
        
        # Initialize the model
        model = NeuralNet(input_size, hidden_size, output_size)
        model.load_state_dict(data["model_state"])
        model.eval()
        
        return model, all_words, tags, spell_checker
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None

def get_response(sentence, model, all_words, tags, spell_checker):
    # Correct spelling in the input sentence
    corrected_sentence = spell_checker.correct_text(sentence)
    print(f"Corrected input: {corrected_sentence}")
    
    # Tokenize and stem the corrected sentence
    sentence_words = tokenize(corrected_sentence)
    sentence_words = [stem(word) for word in sentence_words]
    
    # Create bag of words
    X = bag_of_words(sentence_words, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    
    # Get prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    # Get probabilities
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    # If probability is too low, return a default response
    if prob.item() < 0.5:
        return "I'm not sure I understand. Could you please rephrase your question?"
    
    # Get a random response for the predicted tag
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

def main():
    print("Loading model and data...")
    model, all_words, tags, spell_checker = load_model()
    
    if model is None:
        print("Failed to load model. Please check if the model file exists.")
        return
    
    print("Bot is ready! Type 'quit' to exit")
    print("Note: Your input will be automatically spell-checked")
    
    while True:
        sentence = input("You: ")
        if sentence.lower() == 'quit':
            break
            
        # Get suggestions for misspelled words
        words = sentence.split()
        for word in words:
            if word.lower() not in spell_checker.medical_terms:
                suggestions = spell_checker.get_suggestions(word)
                if suggestions and suggestions[0] != word:
                    print(f"Did you mean '{suggestions[0]}' instead of '{word}'?")
        
        response = get_response(sentence, model, all_words, tags, spell_checker)
        print(f"Bot: {response}")

if __name__ == "__main__":
    # Load intents for responses
    with open('intents.json', 'r') as f:
        intents = json.load(f)
    main() 