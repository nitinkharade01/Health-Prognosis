import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
import torch
import json

class ChatBotApp:
    def __init__(self, master):
        self.master = master
        master.title("ChatBot")

        # Load intents from JSON file
        self.intents_data = self.load_json('intents.json')

        # Load processed results from .pth file
        self.results = self.load_results('processed_data.pth')

        # Create chat history display
        self.chat_history = scrolledtext.ScrolledText(master, width=50, height=20)
        self.chat_history.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

        # Create input box for user query
        self.query_input = tk.Entry(master, width=40)
        self.query_input.grid(row=1, column=0, padx=10, pady=5)

        # Create send button
        self.send_button = tk.Button(master, text="Send", command=self.send_query)
        self.send_button.grid(row=1, column=1, padx=10, pady=5)

    def load_json(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        return data

    def load_results(self, filename):
        results = torch.load(filename)
        return results

    def process_query(self, query):
        max_score = -1
        matched_intent = None
        # Loop through intents and find the best match
        for intent in self.intents_data['intents']:
            for pattern in intent['patterns']:
                # Here you can use your NLP model to calculate similarity scores
                # For simplicity, let's assume exact match for now
                if pattern.lower() == query.lower():
                    # You can also calculate a score based on similarity
                    # Update max_score and matched_intent if needed
                    max_score = 1
                    matched_intent = intent
                    break
        return matched_intent

    def get_response(self, intent):
        # Define responses based on the intent
        # You can customize the responses based on your JSON file
        return intent['responses'][0]

    def send_query(self):
        # Get user query from input box
        user_query = self.query_input.get()
        # Process user query
        intent = self.process_query(user_query)
        # Get response based on the intent
        if intent:
            bot_response = self.get_response(intent)
            # Display bot response in chat history
            self.chat_history.insert(tk.END, "User: {}\n".format(user_query))
            self.chat_history.insert(tk.END, "Bot: {}\n".format(bot_response))
        else:
            messagebox.showinfo("Error", "Sorry, I didn't understand that.")

def main():
    root = tk.Tk()
    app = ChatBotApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


