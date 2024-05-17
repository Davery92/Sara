import base64
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, session
from faster_whisper import WhisperModel
import ollama
import os
import torch
import pyaudio
import wave
from melo.api import TTS
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from datetime import datetime
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, Email
from flask import flash




app = Flask(__name__, template_folder=".")

conversation = []
conversation.append({'role': 'system', 'content': "You are a smart, flirty and witty personal assistant named Sara. You keep your answers short and sweet"})

app.secret_key = 'your secret key'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class SignupForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Signup')
# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# Form for login
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/login', methods=['GET', 'POST'])
def login():
    login_form = LoginForm()
    if login_form.validate_on_submit():
        with open('static/credentials.txt', 'r') as f:
            for line in f:
                if ':' not in line:
                    continue
                username, password = line.strip().split(':')
    # Rest of your code here
                if login_form.email.data == username and login_form.password.data == password:
                    user = User(login_form.email.data)
                    login_user(user)
                    return redirect(url_for('index'))  # Redirect to a protected page after login
            flash('Incorrect username or password')
    return render_template('pages/login.html', login_form=login_form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    signup_form = SignupForm()
    if signup_form.validate_on_submit():
        with open('static/credentials.txt', 'a') as f:
            f.write(f"{signup_form.email.data}:{signup_form.password.data}\n")
        flash('Signup successful. You can now login.')
        return redirect(url_for('login'))  # Redirect to the login page after signup
    return render_template('pages/signup.html', signup_form=signup_form)



@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))  # Redirect to the login page after logout




def save_conversation_to_file(conversation):
    # Get the current date
    now = datetime.now()
    # Format the date as a string
    date_string = now.strftime("%A, %B %d")
    # Create the filename
    filename = f'static/past_conversations/{date_string}.txt'
    with open(filename, 'a') as f:
        message = conversation[-2]  # Get the last message in the conversation
        if message['role'] in ['user', 'assistant']:
            f.write(f"{message['role']}: {message['content']}\n")

model_size = "large-v3"
# Run on GPU with FP16
wmodel = WhisperModel(model_size, device="cuda", compute_type="float16")

def play_audio(file_path):
    wf = wave.open(file_path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    stream.stop_stream()
    stream.close()
    p.terminate()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)



def process_and_play(prompt):
    
    speed = 1.3

    # CPU is sufficient for real-time inference.
    # You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
    device = 'auto' # Will automatically use GPU if available

    # English 
    text = prompt
    model = TTS(language='EN', device=device)
    speaker_ids = model.hps.data.spk2id
    output_path = r'E:\Projects\Sara\dev\static\output.mp3'
    model.tts_to_file(text, speaker_ids['EN-AU'], output_path, speed=speed)
    
    #play_audio('E:\Projects\Sara\outputs\output.wav')

def get_Chat_response(conversation):
    
    stream = ollama.chat(
    model='llama3',
    messages=conversation,
    stream=False,
)
    output = ""
    response = stream['message']['content']
    print(response)
    return response

def whisper_processing(audio_path):
    
    segments, info = wmodel.transcribe(audio_path, beam_size=5, language='en')
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        trans = ""
        trans += segment.text + " "
    trans = trans.strip()
    return trans
@app.route('/about')
def about():
    return render_template('pages/about.html')

@app.route('/recorded_conversations')
def recorded_conversations():
    files = os.listdir('static/recorded_conversations/')
    return render_template('pages/recorded_conversations.html', files=files)

@app.route('/past_conversations')
def past_conversations():
    files = os.listdir('static/past_conversations/')
    return render_template('pages/past_conversations.html', files=files)

@app.route('/delete_rec_file', methods=['POST'])
def delete_rec_file():
    data = request.get_json()
    filename = data.get('filename')
    if filename:
        try:
            os.remove(os.path.join('static/recorded_conversations', filename))
            return '', 200  # Return a success status
        except OSError:
            pass
    return '', 400  # Return an error status if the file couldn't be deleted

@app.route('/delete_conv_file', methods=['POST'])
def delete_conv_file():
    data = request.get_json()
    filename = data.get('filename')
    if filename:
        try:
            os.remove(os.path.join('static/past_conversations', filename))
            return '', 200  # Return a success status
        except OSError:
            pass
    return '', 400  # Return an error status if the file couldn't be deleted

@app.route("/user_input", methods=['POST'])
def handle_user_input():
    try:
        data = request.json['userInput']
        conversation.append({'role': 'user', 'content': data})
        if len(conversation) > 10:
            conversation.pop(0)
        chat_response = get_Chat_response(conversation)
        retry_count = 0
        while chat_response == None  or chat_response == "" and retry_count < 5:
            chat_response = get_Chat_response(conversation)
            retry_count += 1
            print(f"Retry count: {retry_count}")

        if chat_response is None:
            raise Exception("Failed to generate chat response after 3 attempts")
        conversation.append({'role': 'assistant', 'content': chat_response})
        save_conversation_to_file(conversation)
        return jsonify({'chat_response': chat_response})
    except Exception as e:
        app.logger.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

def format_date(dt):
    # Get the day of the month
    day = dt.day
    # Determine the suffix
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = ["st", "nd", "rd"][day % 10 - 1]
    # Format the date
    return dt.strftime(f"%A, %B {day}{suffix}, %Y %I-%M%p")

@app.route("/record_processing", methods=['POST'])
def handle_record_processing():
    try:
        # Load the recorded audio data
        audio_path = r"E:\Projects\Sara\dev\static\rec-audio.wav"
        # Process the audio data with whisper_processing
        processed_audio = whisper_processing(audio_path)
        # Get the current date and time
        now = datetime.now()
        # Format the date and time as a string
        date_time = format_date(now)
        # Create the output file path
        output_path = f"E:\\Projects\\Sara\\dev\\static\\recorded_conversations\\{date_time}.txt"
        with open(output_path, 'w') as f:
            f.write(processed_audio)
        # Save the processed audio data to the output file
        os.remove(audio_path)
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        app.logger.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/reset", methods=['POST'])
def reset_conversation():
    conversation.clear()
    return jsonify({'status': 'success'}), 200

@app.route("/recorded_audio_data", methods=['POST'])
def handle_recorded_audio_data():
    try:
        data = request.json['audioData']
        decode = base64.b64decode(data)

        # Write the incoming audio data to a temporary file
        with open(r"E:\Projects\Sara\dev\static\temp-audio.wav", "wb") as f:
            f.write(decode)

        # Load the existing audio data
        
        existing_audio_path = r"E:\Projects\Sara\dev\static\rec-audio.wav"
        if os.path.exists(existing_audio_path):
            # Load the existing audio data
            existing_audio = AudioSegment.from_wav(existing_audio_path)
        else:
            # Create an empty AudioSegment
            existing_audio = AudioSegment.empty()

        # Load the new audio data
        new_audio = AudioSegment.from_wav(r"E:\Projects\Sara\dev\static\temp-audio.wav")

        # Append the new audio data to the existing audio data
        combined_audio = existing_audio + new_audio

        # Write the combined audio data to the file
        combined_audio.export(r"E:\Projects\Sara\dev\static\rec-audio.wav", format="wav")

        return jsonify({'status': 'success'}), 200
    except Exception as e:
        app.logger.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/audio_data", methods=['POST'])
def handle_audio_data():
    try:
        data = request.json['audioData']
        decode = base64.b64decode(data)

        with open(r"E:\Projects\Sara\audio.wav", "wb") as f:
            f.write(decode)
        audio_path = r"E:\Projects\Sara\audio.wav"
        result = whisper_processing(audio_path)
        conversation.append({'role': 'user', 'content': result})
        if len(conversation) > 10:
            conversation.pop(0)
        chat_response = get_Chat_response(conversation)
        retry_count = 0
        while chat_response == None  or chat_response == "" and retry_count < 5:
            chat_response = get_Chat_response(conversation)
            retry_count += 1
            print(f"Retry count: {retry_count}")

        if chat_response is None:
            raise Exception("Failed to generate chat response after 3 attempts")
        process_and_play(chat_response)
        save_conversation_to_file(conversation)
        return jsonify({'transcription': result, 'chat_response': chat_response})
    except Exception as e:
        app.logger.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/")
@login_required
def index():
    return render_template("pages/home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)


