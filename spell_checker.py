from symspellpy import SymSpell, Verbosity
import json
import pkg_resources

class MedicalSpellChecker:
    def __init__(self):
        # Initialize the spell checker with a dictionary
        self.spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        
        # Load the dictionary
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt")
        self.spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        
        # Load medical terms from intents
        self.medical_terms = set()
        self.load_medical_terms()
        
        # Add medical terms to the dictionary
        for term in self.medical_terms:
            self.spell.create_dictionary_entry(term, 1)
        
    def load_medical_terms(self):
        try:
            with open('intents.json', 'r', encoding='utf-8') as f:
                intents = json.load(f)
                
            # Extract all words from patterns
            for intent in intents['intents']:
                for pattern in intent['patterns']:
                    words = pattern.lower().split()
                    self.medical_terms.update(words)
                    
            # Add common medical terms
            common_medical_terms = {
                'diabetes', 'hypertension', 'cholesterol', 'cardiovascular',
                'respiratory', 'neurological', 'gastrointestinal', 'arthritis',
                'osteoporosis', 'asthma', 'allergy', 'anemia', 'thyroid',
                'vitamin', 'mineral', 'protein', 'carbohydrate', 'fat',
                'exercise', 'diet', 'nutrition', 'therapy', 'medication',
                'symptom', 'diagnosis', 'treatment', 'prevention', 'recovery',
                'wellness', 'health', 'medical', 'doctor', 'nurse', 'patient',
                'hospital', 'clinic', 'pharmacy', 'prescription', 'dosage',
                'side effect', 'allergic', 'immune', 'infection', 'virus',
                'bacteria', 'fungal', 'viral', 'chronic', 'acute', 'severe',
                'mild', 'moderate', 'prognosis', 'rehabilitation', 'surgery',
                'anesthesia', 'antibiotic', 'antiviral', 'vaccine', 'immunization'
            }
            self.medical_terms.update(common_medical_terms)
            
        except Exception as e:
            print(f"Error loading medical terms: {e}")
            self.medical_terms = set()
    
    def correct_text(self, text):
        """
        Correct spelling in the input text while preserving medical terms
        """
        # Split the text into words
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Check if the word is a medical term
            if word.lower() in self.medical_terms:
                corrected_words.append(word)
            else:
                # Get the correction for non-medical terms
                suggestions = self.spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
                if suggestions:
                    corrected_words.append(suggestions[0].term)
                else:
                    corrected_words.append(word)
        
        # Join the words back into a sentence
        return ' '.join(corrected_words)
    
    def get_suggestions(self, word, max_suggestions=3):
        """
        Get spelling suggestions for a word
        """
        if word.lower() in self.medical_terms:
            return [word]  # Return the word itself if it's a medical term
            
        suggestions = self.spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        return [s.term for s in suggestions[:max_suggestions]] 