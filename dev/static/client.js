let audioContext;
let source;
let processor;
let isRecording = false;

document.getElementById('StartChat').addEventListener('click', function() {
    let button = this; // Store a reference to the button
    if (!isRecording) {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(function(stream) {
                audioContext = new window.AudioContext();
                source = audioContext.createMediaStreamSource(stream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);

                source.connect(processor);
                processor.connect(audioContext.destination);

                processor.onaudioprocess = function(e) {
                    // e.inputBuffer is the audio data
                    const xhr = new XMLHttpRequest();
                    xhr.open('POST', '/process_audio', true);
                    xhr.setRequestHeader('Content-Type', 'application/json');
                    xhr.send(JSON.stringify({ audioData: Array.from(e.inputBuffer.getChannelData(0)) }));
                };

                isRecording = true;
                document.getElementById('mic-icon').classList.remove('hidden');
                button.textContent = 'Stop Chat'; // Use the stored reference
            });
    } else {
        // Disconnect and stop the audio processor
        processor.disconnect(audioContext.destination);
        source.disconnect(processor);
        isRecording = false;
        document.getElementById('mic-icon').classList.add('hidden');
        button.textContent = 'Start Chat'; // Use the stored reference
    }
});

document.getElementById('UploadDoc').addEventListener('click', function() {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload_documents', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send(JSON.stringify({}));
});