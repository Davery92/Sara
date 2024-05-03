**Sara: A Voice Chat AI**
==================================

Welcome to Sara, an AI-powered voice chat platform that lets you have conversations like never before. With advanced natural language processing (NLP) and machine learning
capabilities, Sara is designed to understand and respond to your thoughts and emotions in a way that feels human-like.

**Getting Started**
-------------------

To get started with Sara, follow these steps:

### Step 1: Install OpenVoice

OpenVoice is an open-source speech recognition library that provides the foundation for Sara's voice processing capabilities. To install OpenVoice, run the following command:
```
pip install openvoice
```
### Step 2: Download Checkpoints from OpenVoice

After installing OpenVoice, you'll need to download the necessary checkpoints from their GitHub page. These checkpoints are pre-trained models that enable Sara to recognize and
respond to your voice input.

1. Visit the [OpenVoice GitHub page]([https://github.com/OpenVoiceAI/openvoice](https://github.com/myshell-ai/OpenVoice/blob/main/docs/USAGE.md)).
2. Follow the instructions for the V2 option.


### Step 3: Download and Install Vosk Model

Vosk is an open-source speech recognition model that Sara uses for voice-to-text conversion. To download and install the vosk model, follow these steps:

1. Visit their page(https://alphacephei.com/vosk/).
2. Follow their installation instructions.
3. Click on the "models" button.
4. Download the file "vosk-model-small-en-us-0.15".

### Step 4: Install Faster-Whisper

Faster-Whisper is an open-source library that enables Sara to recognize and respond to voice input in real-time. To install Faster-Whisper, run the following command:
```
pip install faster-whisper
```

### Step 5: Create a Python 3.9 Virtual Environment

Sara is built on top of Python 3.9, so we need to create a virtual environment for our project.

```
python3.9 -m venv env
```
### Step 6: Install the Requirements

```
pip install -r requirements.txt
```

**Setting Up Sara**
-------------------

Make sure to have the following folders in the root directory:

1. checkpoints_v2
2. docs
3. env
4. openvoice
5. outputs
6. processed
7. resources
8. vosk-model-small-en-us-0.15
   

**Running Sara**
----------------

To run Sara, simply run
```
python speak.py
```
**Using Your Wake Word**

At this point, feel free to say the demo wake word "sarah" (or whatever wake word you've chosen) to initiate the conversation. To select your own wake word, simply modify the
`speaking.py` file where it says `response_word = "sarah"` and enter your preferred keyword.

**How It Works**

Once Sara recognizes your wake word, she'll respond with "Hotword detected! Recording..." and wait for your input. Speak freely until you're finished, then automatically, Sara
will process the text and generate a response using the LLaMA model (which can be changed within the `speaking.py` file). The generated response is then passed to our TTS
program, which plays it back once generated.

**Conversation Loop**

The conversation loop continues as follows: after the response is played, Sara will display the word "speak" to indicate she's listening again. You'll have 2 seconds to speak
before the system loops back and displays the word "Listening..." again. If you see this message, please say your wake word once more to continue the conversation.

**Tips for Seamless Conversations**

To ensure a seamless conversation experience:

* Make sure to say your wake word within the 2-second time limit.
* Speak clearly and at a comfortable pace.
* Feel free to ask Sara questions or share your thoughts – she's designed to respond thoughtfully and engage in meaningful conversations!

I hope you enjoy using Sara, and I'm always looking for ways to improve her capabilities. Happy chatting!



### Future items to be added

1. Function calling
2. Memory Database
3. Homeassistant integration
4. Note taking
5. Timer functionality
6. Cell Phone Push Alerts

