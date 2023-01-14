# Story Generator

This project is based on AIDungeon [here](https://github.com/Latitude-Archives/AIDungeon/)

## Installation and Run

1. install nvidia-driver
    ```
    sudo apt update
    git clone https://github.com/GoogleCloudPlatform/compute-gpu-installation.git
    cd compute-gpu-installation/linux/
    sudo python3 install-gpu-driver.py
    ```
2. install anaconda and create env

    You should use Python 3.7.9 or older version.
    ```
    conda create -n story_generator python=3.7.9
    ```
3. install cuda 10.0 + cudnn7.5, tensorflow-gpu==1.15.0
    ```
    conda activate story_generator
    conda install cudatoolkit=10.0
    conda install cudnn=7.5.0
    conda install tensorflow-gpu=1.15
    ```
4. Clone the project and install the required dependencies
    ```
    git clone https://github.com/JamesWanglf/AI_Dungeon.git
    cd AI_Dungeon/
    pip install -r requirements.txt
    pip install werkzeug==2.2.2
    ```
5. Download model
    ```
    cd AI_Dungeon/
    sudo su
    ./download_model.sh
    ```
6. Now, you can run the project.
    ```
    python app.py
    ```
    This command will run the API on port 8000.

## API endpoints

1. Initialize the story generator

    You should initialize the story generator before you call the `/generate-story` or `/add-text-to-story` endpoint.

    url: `/init-story-generator`

    method: `POST`

    sample request: 
    ```
    curl --location --request POST 'http://{hostname}/init-story-generator'
    --header 'Content-Type: application/json'
    --data-raw 
    '{
        "prompt": "Your initializing text is here."
    }'
    ```

    sample response: 
    ```
    status_code: 200
    response content: 
    {
        "result": "Generated text ****"
    }
    ```

    error response:
    ```
    status_code: 400
    response content: 
    {
        "error": "Invalid request. The request should contain \"prompt\"."
    }
    ```

2. Generate the context

    This will return 3 candidates generated from the initialing context and the input prompt.

    url: `/generate-story`

    method: `POST`

    sample request:
    ```
    curl --location --request POST 'http://{hostname}/generate-story'
    --header 'Content-Type: application/json'
    --data-raw 
    '{
        "prompt": "Your prompt is here."
    }'
    ```

    sample response:
    ```
    {
        "results": ["text 1", "text 2", ...]
    }
    ```

    error responses:
    ```
    status_code: 400
    {
        "error": "Invalid request. The request should contain \"prompt\"."
    }

    status_code: 500
    {
        "error": "Please initialize the story generator first."
    }
    ```

3. Add the context to the story

    You can select one from the candidates and add it to the story.

    url: `/add-text-to-story`

    method: `POST`

    sample request:
    ```
    curl --location --request POST 'http://{hostname}/add-text-to-story'
    --header 'Content-Type: application/json'
    --data-raw 
    '{
        "text": "Your text is here."
    }'
    ```

    sample response:
    ```
    status_code: 200
    {
        "result": "success"
    }
    ```

    error response:
    ```
    status_code: 400
    {
        "error": "Invalid request. The request should contain \"text\"."
    }

    status_code: 500
    {
        "error": "Please initialize the story generator first."
    }

    status_code: 500
    {
        "result": "failure"
    }
    ```
