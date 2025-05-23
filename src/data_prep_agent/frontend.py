import os
from flask import Flask, request, render_template_string, jsonify
from werkzeug.utils import secure_filename
import requests
import json

UPLOAD_FOLDER = 'data_prep_agent/src/data_files'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>CSV Uploader & Chat</title>
    <style>
        body { font-family: Arial, margin: 40px; }
        #chat-box { border: 1px solid #ccc; height: 300px; overflow-y: scroll; padding: 10px; margin-bottom: 10px; }
        #chat-input { width: 80%; }
        #send-btn { width: 18%; }
    </style>
</head>
<body>
    <h2>Upload CSV Files</h2>
    <form id="upload-form" enctype="multipart/form-data">
        <input type="file" name="files" multiple accept=".csv">
        <button type="submit">Upload</button>
    </form>
    <div id="upload-status"></div>
    <hr>
    <h2>Chat Window</h2>
    <div id="chat-box"></div>
    <input type="text" id="chat-input" placeholder="Type your message...">
    <button id="send-btn">Send</button>
    <script>
        // File upload
        document.getElementById('upload-form').onsubmit = async function(e) {
            e.preventDefault();
            let formData = new FormData(this);
            let res = await fetch('/upload', { method: 'POST', body: formData });
            let data = await res.json();
            document.getElementById('upload-status').innerText = data.message;
        };

        // Chat functionality
        const chatBox = document.getElementById('chat-box');
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');

        function appendMessage(sender, text) {
            let msg = document.createElement('div');
            msg.innerHTML = '<b>' + sender + ':</b> ' + text;
            chatBox.appendChild(msg);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        sendBtn.onclick = async function() {
            let message = chatInput.value.trim();
            if (!message) return;
            appendMessage('You', message);
            chatInput.value = '';
            let res = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });
            let data = await res.json();
            appendMessage('Server', data.reply);
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'message': 'No files part'}), 400
    files = request.files.getlist('files')
    saved = 0
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            saved += 1
    return jsonify({'message': f'{saved} file(s) uploaded successfully.'})

@app.route('/chat', methods=['POST'])
def chat():
    if not os.listdir(app.config['UPLOAD_FOLDER']):
        return jsonify({"reply": "please upload a file first"})

    data = request.get_json()
    user_message = data.get('message', '')
    url = "http://localhost:8000/generate"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "input_message": f"{user_message}",
        "use_knowledge_base": True
    }

    aiq_reply = requests.post(url, headers=headers, data=json.dumps(data))

    return jsonify({"reply": aiq_reply})

if __name__ == '__main__':
    app.run(debug=True)
