from flask import Flask, render_template, request, jsonify
import json
import random
from datetime import datetime
import subprocess
import re

app = Flask(__name__)

# Load the datasets
def load_datasets():
    try:
        with open('message.json', 'r', encoding='utf-8') as file:
            message_data = json.load(file)
        with open('exercises.json', 'r', encoding='utf-8') as file:
            exercises_data = json.load(file)
        with open('mood.json', 'r', encoding='utf-8') as file:
            mood_data = json.load(file)
        print(f"Loaded {len(message_data['intents'])} intents from message.json")
        print(f"Loaded {len(exercises_data['intents'])} intents from exercises.json")
        return message_data, exercises_data, mood_data
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        return None, None

message_data, exercises_data, mood_data = load_datasets()

if message_data is None or exercises_data is None or mood_data is None:
    raise Exception("Failed to load required datasets")

# Initialize user mood tracking
user_mood = {
    'current_mood': 'neutral',
    'mood_score': 5,
    'conversation_keywords': []
}

# Combine all intents for easier searching
all_intents = message_data['intents'] + exercises_data['intents']
print(f"Total intents loaded: {len(all_intents)}")

# Initial assessment questions
initial_questions = [
    "How would you rate your current mood on a scale of 1-10?",
    "Have you been experiencing any changes in your sleep patterns lately?",
    "Are you having trouble concentrating or making decisions?",
    "How is your appetite and have there been any changes recently?",
    "Are you feeling overwhelmed or experiencing high levels of stress?"
]

# Track the current question index for each session
question_index = 0
is_initial_assessment = True

def preprocess_text(text):
    # Convert to lowercase and strip whitespace
    text = text.lower().strip()
    # Remove common punctuation
    text = text.replace('?', '').replace('.', '').replace('!', '')
    return text

def get_word_similarity(str1, str2):
    # Convert strings to sets of words
    set1 = set(str1.split())
    set2 = set(str2.split())
    
    # Calculate intersection and union
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    # Return Jaccard similarity
    return intersection / union if union > 0 else 0

def update_user_mood(message):
    global user_mood
    
    # Update conversation keywords
    user_mood['conversation_keywords'].append(message.lower())
    
    # Check for mood indicators
    for mood_type, data in mood_data['mood_indicators'].items():
        if any(keyword in message.lower() for keyword in data['keywords']):
            user_mood['current_mood'] = mood_type
            # Update mood score (1-10 scale)
            if mood_type == 'positive':
                user_mood['mood_score'] = min(user_mood['mood_score'] + 1, 10)
            elif mood_type == 'sad':
                user_mood['mood_score'] = max(user_mood['mood_score'] - 1, 1)

def get_time_based_greeting():
    current_hour = datetime.now().hour
    
    for period, data in mood_data['time_greetings'].items():
        start_hour, end_hour = data['hours']
        if start_hour <= current_hour < end_hour or (period == 'night' and (current_hour >= start_hour or current_hour < end_hour)):
            return random.choice(data['messages'])
    
    return random.choice(mood_data['time_greetings']['day']['messages'])

def generate_farewell_message():
    mood = user_mood['current_mood']
    motivation = random.choice(mood_data['mood_indicators'][mood]['motivation'])
    time_greeting = get_time_based_greeting()
    
    # Get a relevant exercise based on mood
    relevant_exercise = None
    for intent in exercises_data['intents']:
        if mood in intent['tag'].lower():
            relevant_exercise = random.choice(intent['responses'])
            break
    
    farewell = f"Today your mood seems {mood}, but remember: {motivation}\n"
    if relevant_exercise:
        farewell += f"\nHere's something you can try:\n{relevant_exercise}\n"
    farewell += f"\n{time_greeting}"
    
    return farewell

def check_ollama_status():
    try:
        # Check if Ollama is running and the model is available
        check_process = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        print(f"Ollama status check output: {check_process.stdout}")
        print(f"Ollama status check error: {check_process.stderr}")
        return "gemma3:1b" in check_process.stdout
    except Exception as e:
        print(f"Error checking Ollama status: {str(e)}")
        return False

def get_ollama_response(prompt):
    process = None
    try:
        # First check if Ollama is available
        if not check_ollama_status():
            print("Ollama is not running or gemma3:1b model is not installed")
            return None

        # Add word limit instruction to the prompt
        modified_prompt = f"Please provide a concise response in under 300 words to: {prompt}"
        print(f"Sending prompt to Ollama: {modified_prompt}")
        
        # Run the Ollama model with UTF-8 encoding
        startupinfo = None
        if hasattr(subprocess, 'STARTUPINFO'):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        cmd = ["ollama", "run", "gemma3:1b", modified_prompt]
        print(f"Executing command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
            encoding='utf-8',
            errors='replace'
        )
        print("Process started, waiting for response...")
        output, error = process.communicate()
        
        if process.returncode != 0:
            print(f"Ollama process returned non-zero exit code: {process.returncode}")
            print(f"Ollama error output: {error}")
            return None
            
        if error:
            print(f"Ollama stderr output: {error}")
            
        # Clean and process the response
        response = output.strip()
        print(f"Raw Ollama response: {response[:100]}...") # Print first 100 chars
        
        if response:
            # Split into words and limit to 300 words
            words = response.split()
            if len(words) > 300:
                words = words[:300]
                response = ' '.join(words) + '...'
            print(f"Processed response length: {len(words)} words")
            return response
            
        print("No response received from Ollama")
        return None
    except Exception as e:
        print(f"Error calling Ollama: {str(e)}")
        print(f"Exception type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None
    finally:
        if process and process.poll() is None:
            try:
                process.kill()
                process.communicate()  # Clean up pipes
                print("Cleaned up Ollama process")
            except Exception as e:
                print(f"Error cleaning up process: {str(e)}")

def get_best_match(user_input):
    # Check for farewell messages
    farewell_keywords = {'bye', 'goodbye', 'see you', 'talk to you later', 'until next time'}
    if any(keyword in user_input.lower() for keyword in farewell_keywords):
        return generate_farewell_message()
    
    # Preprocess user input
    processed_input = preprocess_text(user_input)
    print(f"Processing user input: {processed_input}")  # Debug log
    
    # Update user mood based on input
    update_user_mood(user_input)
    
    best_match = None
    highest_similarity = 0
    
    # Keywords for sleep-related issues
    sleep_keywords = {'sleep', 'slept', 'sleeping', 'insomnia', 'bed', 'night', 'rest', 'tired', 'dreams'}
    
    # Search through all intents
    for intent in all_intents:
        # Check if this is a sleep-related intent
        is_sleep_intent = 'sleep' in intent['tag'].lower()
        
        # Convert and preprocess patterns
        patterns = [preprocess_text(pattern) for pattern in intent['patterns']]
        
        # Check for exact matches first
        if processed_input in patterns:
            print(f"Found exact match in tag: {intent['tag']}")
            return random.choice(intent['responses'])
        
        # Check each pattern
        for pattern in patterns:
            # Calculate word-based similarity
            similarity = get_word_similarity(processed_input, pattern)
            
            # Boost similarity for sleep-related patterns if input contains sleep keywords
            if is_sleep_intent and any(word in processed_input for word in sleep_keywords):
                similarity += 0.2
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = intent
                print(f"Found match in tag: {intent['tag']} with similarity: {similarity}")
    
    # Lower threshold for sleep-related queries
    threshold = 0.3 if any(word in processed_input for word in sleep_keywords) else 0.4
    
    # If we found a good match
    if best_match and highest_similarity > threshold:
        return random.choice(best_match['responses'])
    
    # If no match is found in datasets, try Ollama
    print("No match found in datasets, trying Ollama")
    ollama_response = get_ollama_response(user_input)
    
    if ollama_response:
        return ollama_response
    
    # Default response if both dataset and Ollama fail
    print("No response from Ollama either")
    return "I understand you might be having trouble with sleep. Can you tell me more about your sleep patterns? For example, are you having trouble falling asleep, staying asleep, or waking up too early?" if any(word in processed_input for word in sleep_keywords) else "I'm sorry, I couldn't understand that properly. Could you rephrase your question or concern?"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat')
def chat():
    global question_index, is_initial_assessment, user_mood
    # Reset the question index, assessment flag, and user mood for new chat session
    question_index = 0
    is_initial_assessment = True
    user_mood = {
        'current_mood': 'neutral',
        'mood_score': 5,
        'conversation_keywords': []
    }
    return render_template('chat.html', first_question=initial_questions[0], initial_questions=initial_questions)

@app.route('/get_response', methods=['POST'])
def get_response():
    global question_index, is_initial_assessment
    user_message = request.json.get('message', '')
    
    if not user_message:
        return jsonify({'response': "I didn't catch that. Could you please say it again?"})
    
    print(f"\n--- Processing new message: {user_message} ---")  # Debug log
    
    if is_initial_assessment:
        # Store the user's answer (you might want to save these answers in a database)
        print(f"Question {question_index + 1} answer: {user_message}")
        
        # Move to next question or finish assessment
        question_index += 1
        if question_index < len(initial_questions):
            return jsonify({'response': initial_questions[question_index]})
        else:
            is_initial_assessment = False
            question_index = 0
            return jsonify({
                'response': "Thank you for answering these questions. Now, how can I help you today?",
                'assessment_complete': True
            })
    
    # Normal conversation flow
    response = get_best_match(user_message)
    print(f"Final response: {response}\n")  # Debug log
    return jsonify({'response': response})

@app.route('/get_initial_questions')
def get_initial_questions():
    return jsonify({'questions': initial_questions})

if __name__ == '__main__':
    app.run(debug=True)
