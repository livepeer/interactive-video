import base64
import json
import requests


def test_load_model():
    url = 'http://127.0.0.1:5000/load_model'
    headers = {
        'Content-Type': 'application/json'
    }

    data = {
        'language': 'en',
        'model_size': 'tiny',   # 【'tiny', 'base', 'small', 'medium', 'large']
    }
    
    res = requests.post(url, headers=headers, data=json.dumps(data))

    print(res.text)


def test_transcribe(file_ext='mp3'):
    url = 'http://127.0.0.1:5000/transcribe'
    headers = {
        'Content-Type': 'application/json'
    }

    base64_audio = ''
    if file_ext == 'mp3':
        file_path = 'samples/sample-0.mp3'
    else:
        file_path = 'samples/sample-0.aac'

    with open(file_path, 'rb') as audio_file:
        base64_audio = base64.b64encode(audio_file.read()).decode('utf-8')

    data = {
        'language': 'en',
        'model_size': 'tiny',   # 【'tiny', 'base', 'small', 'medium', 'large']
        'file_content': f'{base64_audio}',
        'file_ext': file_ext
    }
    
    res = requests.post(url, headers=headers, data=json.dumps(data))

    print(res.text)


if __name__ == '__main__':
    test_load_model()
    test_transcribe('mp3')
    test_transcribe('aac')
