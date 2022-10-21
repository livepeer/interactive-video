import io
import json
import numpy as np
import os
import insightface
import threading
from datetime import datetime
from flask import Flask, request, Response
from PIL import Image
from urllib.parse import urlparse, parse_qs


app = Flask(__name__)
model = None


def load_model():
    """
    Load face recognition model
    """
    global model

    # Method-2, load model directly
    recognition_model_path = os.path.abspath(os.path.join(os.curdir, 'models', 'ms1mv3_arcface_r50_fp16.onnx'))
    model = insightface.model_zoo.get_model(recognition_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    model.prepare(ctx_id=0, input_size=(640, 640))


def extract_feature(face_img):
    """
    Extract the feature vector for the input image
    """
    global model

    if model is None:
        load_model()

    features = model.get_feat([face_img])

    return features[0]


def get_face_feature(face_img, meta_data, face_features):
    """
    Get the feature vector from the face image
    This method will be run in parallel by threads
    """
    start_time = datetime.now()

    # Process the face, get the representation of the detected face
    face_feature = {
        'id': meta_data['id']
    }

    try:        
        # Extract the feature vector
        face_img_representation = extract_feature(face_img)

        # Need to cast the nparray to list
        vector_casted = face_img_representation.tolist()

        face_feature['vector'] = vector_casted
        
    except Exception as e:
        print(e)
        face_feature['vector'] = []

    # Append new feature to Array
    face_features.append(face_feature)

    print((datetime.now() - start_time).total_seconds() * 1000)


@app.route('/', methods=['GET'])
def welcome():
    """
    Welcome page
    """
    return Response("<h1 style='color:blue'>Feature extraction server is running!</h1>", status=200)


@app.route('/extract_feature', methods=['POST'])
def process_detected_faces():
    """
    All requests from the face detection node will be arrived at here
    """
    # Parse request
    body_data = request.form.getlist('face_data')
    face_data = json.loads(body_data[0])
    base_img_list = request.files.getlist('image')

    face_features = []
    threading_pool = []
    for i in range(len(face_data)):
        meta_data = face_data[i]
        base_img = base_img_list[i].read()  # this is bytes data of detected face image

        # Convert the bytes to numpy array
        face_img = np.asarray(Image.open(io.BytesIO(base_img)))

        # Extract feature
        th = threading.Thread(target=get_face_feature, args=(face_img, meta_data, face_features))
        threading_pool.append(th)
        th.start()
        
    for th in threading_pool:
        th.join()
	
    return Response(json.dumps(face_features), status=200)


@app.route('/load_model', methods=['GET'])
def load_model_request():
    """
    Load Model
    """
    load_model()
    return Response("<h1 style='color:blue'>Feature extraction model has been loaded.</h1>", status=200)
    