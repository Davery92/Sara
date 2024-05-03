
import ollama
from vosk import Model, KaldiRecognizer
import json
import wave
from faster_whisper import WhisperModel
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import os
import soundfile as sf
import wave
import pyaudio
import time
import numpy as np
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from vosk import SetLogLevel
SetLogLevel(-1)
from melo.api import TTS



#Setup TTS
ckpt_converter = r'E:\Projects\Sara\checkpoints_v2\converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = r'E:\Projects\Sara\outputs'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)
speed = 1.0
reference_speaker = r'E:\Projects\Sara\resources\temp.mp3' # This is the voice you want to clone
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)

def get_Chat_response(conversation):
    
    stream = ollama.chat(
    model='llama3',
    messages=conversation,
    stream=False,
)
    output = ""
    response = stream['message']['content']
    return response



def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()
    
conversation = []
conversation.append({'role': 'system', 'content': "You are a smart, flirty and witty personal assistant named Sara. You keep your answers short and concise. If there's something you don't know you say I don't know, you do not make up an answer."})


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
    
    texts = {'EN_NEWEST': prompt,
    }
    src_path = f'{output_dir}/tmp.wav'
    for language, text in texts.items():
        model = TTS(language=language, device=device)
        speaker_ids = model.hps.data.spk2id
        
        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            speaker_key = speaker_key.lower().replace('_', '-')
            
            source_se = torch.load(r'E:\Projects\Sara\checkpoints_v2\base_speakers\ses\en-newest.pth', map_location=device)
            model.tts_to_file(text, speaker_id, src_path, speed=speed)
            save_path = f'{output_dir}/output.wav'

            # Run the tone color converter
            encode_message = "@MyShell"
            tone_color_converter.convert(
                audio_src_path=src_path, 
                src_se=source_se, 
                tgt_se=target_se, 
                output_path=save_path,
                message=encode_message)
    play_audio('E:\Projects\Sara\outputs\output.wav')

def whisper_processing():
    segments, info = wmodel.transcribe(r"E:\Projects\Sara\recording.wav", beam_size=5, language='en')
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        trans = ""
        trans += segment.text + " "
        trans = trans.strip()
        return trans

# define the hotword and its confidence threshold
def mic():
    response_word = "sarah"
    #instruction = "computer"
    confidence_threshold = 0.8
    model = Model(r"E:\Projects\Sara\vosk-model-small-en-us-0.15") # path to your VOSK model
    rec = KaldiRecognizer (model, 44100)

    # start listening for the hotword

    #print("Listening...")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True,
    frames_per_buffer=8000)
    print("Listening...")
    while True:
        
        
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True,frames_per_buffer=8000)
        data = stream.read(8000)
        if len (data) == 0:
            break 
        if rec.AcceptWaveform(data):
            res = rec.Result()
            if "text" in res:
                text = json.loads(res) ["text"]
                if response_word in text: # hotword detected with high confidence
                    print("Hotword detected! Recording...")
                    #flag = True
                    while True:
                        


                        p = pyaudio.PyAudio()
                        stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
                        # Define the silence threshold (adjust to your liking)
                        SILENCE_THRESHOLD = 0.05

                        # Initialize a flag to track whether you're speaking or not
                        speaking = False

                        # Initialize a counter for silence duration
                        silence_duration = 0

                        # Initialize a list to hold the audio data
                        frames = []
                        while True:
                            # Read audio data from the stream
                            data = stream.read(1024)
                            np_data = np.frombuffer(data, dtype=np.int16)

                            # Apply amplitude normalization (optional)
                            normalized_data = np_data / 32768.0

                            # Calculate the RMS energy of the signal
                            rms_energy = np.sqrt(np.mean(normalized_data**2))

                            # Check if you're speaking or not based on the RMS energy
                            if rms_energy > SILENCE_THRESHOLD:
                                speaking = True
                                silence_duration = 0
                                frames.append(data)
                            else:
                                speaking = False
                                silence_duration += 1

                            # If you're not speaking for more than 2 seconds, stop recording and break out of the loop
                            if not speaking and silence_duration > 2 * (44100 / 1024):
                                stream.stop_stream()
                                break
                        stream.close()
                        p.terminate()

                        # Save the audio data to a .wav file
                        wf = wave.open(r'E:\Projects\Sara\recording.wav', 'wb')
                        wf.setnchannels(1)
                        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(44100)
                        wf.writeframes(b''.join(frames))
                        wf.close()
                        
                        #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
                        
                        flag = False
                        
                        whisp_trans = whisper_processing()
                        if whisp_trans==None:
                            print("Listening...")
                            break
                        
                            
                        conversation.append({"role": "user", "content": whisp_trans})
                        sara = get_Chat_response(conversation)
                        {"role": "assistant", "content": sara}
                        

                        print(sara)
                        #print(conversation)
                        process_and_play(sara)
                        time.sleep(.5)
                        print("Speak")
                        flag = True
                        
                        if flag == False:
                            print("Listening...")
                            break



                    
                

if __name__ == "__main__":
    while True:
      mic()
