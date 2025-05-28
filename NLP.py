import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def preprocess_text(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    return inputs

def process_text(text, model):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    
    # Assuming you want to use the last layer's hidden states
    hidden_states = outputs.hidden_states[-1]  
    
    return hidden_states

def save_results(filename, results, input_size, hidden_size, output_size, all_words, model_state):
    # Save results, input_size, hidden_size, output_size, all_words, and model_state
    data = {
        'results': results, 
        'input_size': input_size, 
        'hidden_size': hidden_size, 
        'output_size': output_size, 
        'all_words': all_words, 
        'model_state': model_state
    }
    torch.save(data, filename)
    print("Processed NLP results saved to", filename)


def main(model):
    filename = "intents.json"  
    data = load_json(filename)
    
    # Assuming the text is stored under the key 'intents' in the JSON file
    intents = data['intents']
    
    # Concatenate all text data under 'intents' key
    text = ""
    for intent in intents:
        text += " ".join(intent['patterns']) + " "
    
    hidden_states = process_text(text, model)
    
    # Assuming the last layer of hidden states is used
    last_hidden_state = hidden_states
    
    # Assuming the shape of the last hidden state tensor is [batch_size, seq_length, hidden_size]
    # This might not be the case depending on the model and input
    batch_size, seq_length, hidden_size = last_hidden_state.shape
    
    # Input size might be the same as hidden size, depending on the model
    input_size = hidden_size
    
    # Output size might be the same as input size, depending on the model
    output_size = input_size
    
    # Generate all_words from the processed text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    all_words = tokenizer.tokenize(text)
    
    # Extract model state
    model_state = model.state_dict()
    
    output_filename = "processed_results1.pth"  
    save_results(output_filename, hidden_states, input_size, hidden_size, output_size, all_words, model_state)

if __name__ == "__main__":
    # Create the model object before calling the main function
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_hidden_states=True)
    main(model)



# import torch
# from transformers import BertTokenizer, BertForSequenceClassification
# import json

# def load_json(filename):
#     with open(filename, 'r') as file:
#         data = json.load(file)
#     return data

# def preprocess_text(text):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
#     return inputs

# def process_text(text):
#     model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_hidden_states=True)
#     inputs = preprocess_text(text)
#     outputs = model(**inputs)
#     return outputs

# def save_results(filename, results, input_size, hidden_size, output_size, all_words, model_state):
#     # Save results, input_size, hidden_size, output_size, all_words, and model_state
#     data = {'results': results, 'input_size': input_size, 'hidden_size': hidden_size, 'output_size': output_size, 'all_words': all_words, 'model_state': model_state}
#     torch.save(data, filename)
#     print("Processed NLP results saved to", filename)

# def main(model):
#     filename = "intents.json"  
#     data = load_json(filename)
    
#     # Assuming the text is stored under the key 'intents' in the JSON file
#     intents = data['intents']
    
#     # Concatenate all text data under 'intents' key
#     text = ""
#     for intent in intents:
#         text += " ".join(intent['patterns']) + " "
    
#     results = process_text(text)
    
#     # Calculate input size, hidden size, and output size based on the shape of the input and output tensors
#     input_size = results.logits.shape[-1]
#     hidden_size = results.hidden_states[-1].shape[-1]  # This line might throw an error if hidden_states is None
#     output_size = results.logits.shape[-1]
    
#     # Generate all_words from the processed text
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     all_words = tokenizer.tokenize(text)
    
#     # Extract model state
#     model_state = model.state_dict()
    
#     output_filename = "processed_results1.pth"  
#     save_results(output_filename, results, input_size, hidden_size, output_size, all_words, model_state)


# if __name__ == "__main__":
#     # Create the model object before calling the main function
#     model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_hidden_states=True)
#     main(model)
