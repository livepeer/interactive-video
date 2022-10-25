# whisper_api
## Prerequisite
1. Install Nvidia drivers.
2. Install CUDA toolkit 10.2.
3. Install cudnn8.2.1.
4. Install ffmpeg

## Installation
1. ```conda create -n whisper_api python=3.10```
2. ```conda activate whisper_api```
3. ```sudo apt install portaudio19-dev python-pyaudio python3-pyaudio```
4. ```pip install -r requirements.txt```

## Run
```gunicorn -w <number of workers> -b <host:port> --timeout 300 wsgi:app```

### Endpoints
1. load model  
  endpoints: **`/load_model`**  
  method: **`POST`**  
  options:  
    `language`(optional): the language of the audio speech. `en` as default.  
    `model_size`(optional): size of model which will be used to transcribe. candidates will be [`tiny`, `base`, `small`, `medium`, `large`].  
    
    example:
    ```
    curl --location --request POST 'http://localhost:5000/load_model' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "language": "en",
        "model_size": "base"
    }'
    ```
    
2. transcribe the audio content  
  endpoints: **`/transcribe`**  
  method: **`POST`**  
  options:  
    `file_ext`(required): file extention, `mp3` and `aac` format could be received.  
    `file_content`(required): base64 encoded audio file content.  
    `language`(optional): the language of the audio speech. `en` as default.  
    `model_size`(optional): size of model which will be used to transcribe. candidates will be [`tiny`, `base`, `small`, `medium`, `large`].  
    
    If you don't set `language` and `model_size`, it will use the preloaded model.
    
    example:
    ```
    curl --location --request POST 'http://localhost:5000/transcribe' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "file_ext": "mp3",
        "file_content": "xxxxxxxxxxxxx..."
    }'
    ```
