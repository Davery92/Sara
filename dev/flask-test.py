import base64
import whisper
from flask import Flask, render_template, request, jsonify, Response
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




app = Flask(__name__, template_folder=".")

conversation = []
conversation.append({'role': 'system', 'content': "You are a smart, flirty and witty personal assistant named Sara. You keep your answers short and sweet"})

def save_conversation_to_file(conversation):
    with open('conversation.txt', 'a') as f:
        for message in conversation:
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
        date_time = now.strftime("%Y-%m-%d_%H-%M")
        # Create the output file path
        output_path = f"E:\\Projects\\Sara\\dev\\static\\{date_time}.txt"
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
        return jsonify({'transcription': result, 'chat_response': chat_response})
    except Exception as e:
        app.logger.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/")
def index():
    return render_template("index1.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)


