import torch
import json
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BertChatbot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.label_mapping = None
        self.reverse_label_mapping = None
        self.intents = None
        self.load_model()
        
    def load_model(self):
        try:
            # Load the saved model and data
            checkpoint = torch.load('bert_chatbot_model.pth', map_location=self.device)
            
            # Load label mapping
            with open('label_mapping.json', 'r') as f:
                self.label_mapping = json.load(f)
            self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
            
            # Load intents for responses
            with open('intents.json', 'r') as f:
                self.intents = json.load(f)
            
            # Initialize model and tokenizer
            model_name = "dmis-lab/biobert-base-cased-v1.1"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.label_mapping)
            )
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print("Model loaded successfully!")
            print(f"Accuracy: {checkpoint['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def get_response(self, text):
        # Tokenize input
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move inputs to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        # Get predicted intent
        predicted_tag = self.reverse_label_mapping[predicted_idx]
        
        # If confidence is too low, return a default response
        if confidence < 0.5:
            return "I'm not sure I understand. Could you please rephrase your question?"
        
        # Get a random response for the predicted intent
        for intent in self.intents['intents']:
            if intent['tag'] == predicted_tag:
                return random.choice(intent['responses'])
        
        return "I'm not sure how to respond to that."

def main():
    print("Initializing BERT chatbot...")
    chatbot = BertChatbot()
    
    print("\nChatbot is ready! Type 'quit' to exit")
    print("Note: This version uses BioBERT for better understanding of medical terms")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        
        response = chatbot.get_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main() 