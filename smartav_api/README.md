# flask_server
## Prerequisite
1. Install Nvidia drivers on ubuntu-18.04 machine.
2. Install CUDA toolkit 10.2.
3. Install cudnn8.2.1.
4. Install PostgreSQL
   - After install PostgreSQL, please set password for default user 'postgresql'. Otherwise, you can create a new user with password.
   - Create database 'FaceRecognition'. You can use the name what you want.  
   This user credential and database name will be used later.
   - Please allow the remote access to PostgreSQL on ubuntu.
## Installation
1. ```conda create -n flask_server python=3.8```
2. ```conda activate flask_server```
3. ```pip install --upgrade pip```
4. ```sudo apt-get install libpq-dev```
5. ```pip install -r requirements.txt```
  Please pay attention to the version of onnxruntime-gpu. Please install suitable version of onnxruntime according to the versions of cuda and cudnn,you can find the table by the link: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html. If you go through this guide, you can ignore this attention.
6. ```pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html```

## Download models
1. For face detection node, please download the model.
   ```
   mkdir ~/.insightface
   mkdir ~/.insightface/models
   cd ~/.insightface/models
   ```  
   Download the model file from [here](https://drive.google.com/file/d/1_RbGpfrPbgDT8MiY0FTMkP8bor33OGmq/view?usp=sharing), and unzip it.
   If you can find *.onnx files in ~/.insightface/models/buffalo_s/, it's okay.
   
2. For feature extraction node, please download the model.
    ```
    cd face_recognition/feature_extraction
    mkdir models
    gdown https://drive.google.com/uc?id=1py6MWvxugYBK-4YDNNdby955nZf-hjdN -O model_base.pth -O ms1mv3_arcface_r50_fp16.onnx
    ```

3. For image-captioning module, please download the model.
    ```
    cd image_captioning/
    mkdir checkpoints
    cd checkpoints/
    gdown https://drive.google.com/uc?id=1JMmqsL7Nrq4B2WUXt6it3-c-4LnVOthz -O model_base.pth
    ```
4. For generating-question module, please download word vectors.
    ```
    wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
    tar -xvf  s2v_reddit_2015_md.tar.gz
    ```
    ```
    python -m nltk.downloader universal_tagset
    ```
    ```
    cd ~
    gdown "1txK_dE7PkPNy6hBLGD6NppF93hty5QKE&confirm=t"
    unzip nltk_data.zip
    ```
5. For instance-segmentation module, please download models.
    ```
    cd instance_segmentation/
    mkdir weights
    cd weights
    gdown https://drive.google.com/uc?id=1ts3-PSoB9OKi_XmV6uNvARstT93SbPs_ -O yolact_im700_54_800000.pth
    gdown https://drive.google.com/uc?id=1MB9IRrvDvkPkpkdif17eVVT0zBoRBVxj -O yolact_base_54_800000.pth
    gdown https://drive.google.com/uc?id=15SJ3NlIGWjppG3atS6e1EtXuKVFQIvKk -O yolact_resnet50_54_800000.pth
    ```
 
## Run
### 1. Initialize the database
This will create a table named by 'sample_face_vectors' in PostgreSQL database. The database should be created in PostgreSQL first.
```
cd face_recognition/face_detection
python init_db.py -H <db address> -P <db port> -d <db name> -u <username> -p <password>
```  
E.g.  
```
python init_db.py -H 127.0.0.1 -P 5432 -d FaceRecognition -u postgres -p postgres
```
*Attention: While the face detction server is running on any nodes, you can not initialize the database again.*

### 2. Run Feature Extraction Nodes
This app is responsible for the feature extraction from the input image data.  
```cd face_recognition/feature_extraction```  
```gunicorn -w <number of processes> -b 127.0.0.1:8000 wsgi:app```

### 3. Run Main Server
This app is responsible for configuration of database, the face detection, save sample face features to database and comparation between the face feature with sample data.  
Move to flask_server directory.    
```gunicorn -w <number of processes> -b 0.0.0.0:5000 wsgi:app```

### 4. Swagger Documentation for API endpoints
After you run the `Main Server` successfully, you can find the API endpoints from Swagger Documentation.  
Access `http://0.0.0.0:5000/api/docs` to see the documentation.


## Endpoints
### Face Recognition
* /face-recognition/config-database  
    ```
    curl --location --request GET 'http://0.0.0.0:5000/face-recognition/config-database' \
    --header 'Content-Type: application/json' \
    --data-raw '{ \
        "host": <host domain/ip>, \
        "port": <db port>, \
        "db_name": <db name>, \
        "username": <username>, \
        "password": <password> \
    }'
    ```
* /face-recognition/clear-samples  
    ```
    curl --location --request GET 'http://0.0.0.0:5000/face-recognition/clear-samples'
    ```
* /face-recognition/update-samples  
    ```
    curl --location --request POST 'http://0.0.0.0:5000/face-recognition/update-samples' \
    --header 'Content-Type: application/json' \
    --data-raw '[ \
    { \
        "id": "person1", \
        "name: "person1", \
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD....", \
        "metadata": "mydomain.com/myobject1", \
        "action": "embedlink" \
    }, \
    { \
        "id": "person2", \
        "name: "person2", \
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD....", \
        "metadata": "mydomain.com/myobject2", \
        "action": "embedlink" \
    }]'
    ```
    When you send an image for existing "id", the image will be appended to the "id" face.  
    In this way, each face can have multiple feature vectors, and all vectors will be used when we recognize the face.  
    This will increase the accuracy of our recognition module.  
    For example, after you send the above request, send the following request.
    ```
    curl --location --request POST 'http://0.0.0.0:5000/face-recognition/update-samples' \
    --header 'Content-Type: application/json' \
    --data-raw '[{ \
        "id": "person1", \
        "image": "data:image/jpeg;base64,/8a/AgWAGojgegWEGgojEGOIJ....", \
        "name: "", \
        "metadata": "", \
        "action": "" \
    }]'
    ```
    As a result, "person1" will have 2 feature vectors totally. 
    
* /face-recognition/update-metadata/<sample_id>  
   ```
   curl --location --request POST 'http://0.0.0.0:5000/face-recognition/update-metadata/unknown_person_1' \
   --header 'Content-Type: application/json' \
   --data-raw '{ \
      "name": <name>, \
      "metadata": <metadata> \
   }'
   ```
* /face-recognition  
    ```
    curl --location --request POST 'http://0.0.0.0:5000/face-recognition' \
    --header 'Content-Type: application/json' \
    --data-raw '{ \
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD....", \
        "min_distance": 0.35 \
     }'
    ```
    "min_distance" is an optional field. The value of this field is related with the feature extraction model. For example, it will be in [0, 1]. Since we are using "facenet" model as default, the ideal threshold is 0.35.


### Image Captioning
* /image-captioning  
    ```
    curl --location --request POST 'http://0.0.0.0:5000/image-captioning' \
    --header 'Content-Type: application/json' \
    --data-raw '{ \
      "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD....", \
    }
    ```

### Story Generator
[/set-story-gen-api-host](#set-story-gen-api-host)  
[/init-story-generator](#init-story-generator)  
[/set-story-gen-interval](#set-story-gen-interval)  
[/get-generated-story](#get-generated-story)  
[/add-text-to-story](#add-text-to-story)  


#### <a name="set-story-gen-api-host"/>/set-story-gen-api-host  
You should set the host url of the story generator node.

sample request:

```
curl --location --request POST 'http://0.0.0.0:5000/set-story-gen-api-host' \
--header 'Content-Type: application/json' \
--data-raw '{ \
    "hostname": "http://localhost:8000" \
}
```

sample response:
```
status_code: 200
response content:
{
    "result": "success"
}
```

error response:
```
status_code: 400
response content:
{
    "error": "Invalid request."
}
```

#### <a name="init-story-generator"/> /init-story-generator  
You should initialize the story generator before you call the other endpoints related to story generator.

sample request:
```
curl --location --request POST 'http://0.0.0.0:5000/init-story-generator' \
--header 'Content-Type: application/json' \
--data-raw '{ \
    "prompt": "Your initial text" \
}
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

#### <a name="set-story-gen-interval"/> /set-story-gen-interval  
sample request:
```
curl --location --request POST 'http://0.0.0.0:5000/set-story-gen-interval' \
--header 'Content-Type: application/json' \
--data-raw '{ \
    "interval": 3 \
}
```
This means that it will generate the story every time when the `/image-captioning` endpoint has been called 3 times.

sample response:
```
{
    "result": "success"
}
```
error response:
```
{
    "error": "Invalid request."
}
```

#### <a name="get-generated-story"/> /get-generated-story  

This endpoint returns the 3 most recently created stories.

sample request:
```
curl --location --request GET 'http://0.0.0.0:5000/get-generated-story'
```

sample response:
```
{
    "results": ["text 1", "text 2", "text 3]
}
```

#### <a name="add-text-to-story"/> /add-text-to-story  

You can select one from the candidates and add it to the story.

sample request:
```
curl --location --request POST 'http://{hostname}/add-text-to-story' \
--header 'Content-Type: application/json' \
--data-raw \
'{ \
    "text": "Your text is here." \
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

### Generating Questions
* /generate-questions  
    ```
    curl --location --request POST 'http://0.0.0.0:5000/generate-questions' \
    --header 'Content-Type: application/json' \
    --data-raw '{ \
      "content": "xxx", \
      "max_questions": 3 \
    }
    ```
    "max_questions" is the optional field. 4 as default.  
    The response looks like:  
    ```
    {
      "questions": [
        {
            "answer": "this is the correct answer",
            "context": "the context that is used to generate this question",
            "extra_options": [
               "extra_answer_1",
               "extra_answer_2",
               ...
               "extra_answer_n"
            ],
            "id": 1,
            "options": [
               "answer_1",
               "answer_2",
               "answer_3"
            ],
            "options_algorithm": "sense2vec",
            "question_statement": "question",
            "question_type": "MCQ"
        }
      ],
      "statement": <this returns the "content" value in the request.>,
      "time_taken": <time taken in api server>
    }
    ```

### Instance Segmentation
* /instance-segmentation/load-model  
    ```
    curl --location --request POST 'http://0.0.0.0:5000/instance-segmentation/load-model' \
    --header 'Content-Type: application/json' \
    --data-raw '{ \
        "weights": "instance_segmentation/yolov7-seg.pt", \
        "device": 0, \
        "dataset": "instance_segmentation/data/coco.yml", \
        "half": true, \
        "dnn": true \
    }'
    ```  
    All fields in payload are optional.
* /instance-segmentation/detect-objects  
    ```
    curl --location --request POST 'http://0.0.0.0:5000/instance-segmentation/detect-objects' \
    --header 'Content-Type: application/json' \
    --data-raw '{ \
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD....", \
        "top_k": <max number of detected objects>, \
        "score_threshold": 0.5 \
    }'
    ```
    "score_threshold" should be a value between 0 and 1.
