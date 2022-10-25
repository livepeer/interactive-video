import base64
import os
import tempfile
import flask
from flask import request, Response
from flask_cors import CORS
import whisper

app = flask.Flask(__name__)
CORS(app)
SUPPORTED_AUDIO_FORMATS = ['mp3', 'aac']

preloaded_model = {
    'language': 'en',
    'model_size': 'tiny',
    'model': None
}


def load_model(language, model):
    global preloaded_model

    if preloaded_model['model'] is not None and preloaded_model['language'] == language and preloaded_model['model_size'] == model:
        return True

    try:
        preloaded_model['language'] = language
        preloaded_model['model_size'] = model

        if model != 'large' and language == 'english':
            model = model + '.en'

        preloaded_model['model'] = whisper.load_model(model)

        return True

    except Exception as e:
        print(f'Load model failed: {e}')
        return False


@app.route('/load_model', methods=['POST'])
def reload_model():
    global preloaded_model
    
    if request.method == 'POST':
        # parsing request body
        data = request.json

        language = preloaded_model['language']
        if 'language' in data:
            language = data['language']

        model = preloaded_model['model_size']
        if 'model_size' in data:
            model = data['model_size']

        res = load_model(language=language, model=model)

        if res:
            return Response('Model has been loaded successfully', status=200)
        else:
            return Response('Load model failed', status=500)

    else:
        return Response('Invalid request. This endpoint only processes POST request', status=400)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    global preloaded_model

    if request.method == 'POST':
        # parsing request body
        data = request.json
        language = preloaded_model['language']
        if 'language' in data:
            language = data['language']

        model = preloaded_model['model_size']
        if 'model_size' in data:
            model = data['model_size']
        
        if 'file_content' not in data or 'file_ext' not in data:
            return Response('Invalid request', status=400)
        file_ext = data['file_ext']
        encoded_audio_string = data['file_content']

        if file_ext not in SUPPORTED_AUDIO_FORMATS:
            return Response('Invalid request. This endpoint only supports mp3 or aac format', status=400)

        # there are no english models for large
        if preloaded_model['model'] is None:
            res = load_model(language, model)
            if not res:
                return Response('Load model failed', status=500)

        # save audio file in temp directory
        temp_dir = tempfile.mkdtemp()
        save_path = os.path.join(temp_dir, f'temp.{file_ext}')
        with open(save_path, 'wb') as audio_file:
            audio_file.write(base64.b64decode(encoded_audio_string))

        # transcribe
        if preloaded_model['model'] is None:
            return Response('Model is not loaded.', status=500)

        if language == 'english':
            result = preloaded_model['model'].transcribe(save_path, language='english')
        else:
            result = preloaded_model['model'].transcribe(save_path)

        return Response(result['text'], status=200)

    else:
        return Response('Invalid request. This endpoint only processes POST request', status=400)


if __name__ == '__main__':
    app.run()