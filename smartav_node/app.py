import torch
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, Response, jsonify, make_response
from face_recognition.face_detection import main as face_detection_main
from face_recognition.face_detection.common import set_env
from image_captioning import image_captioning
from instance_segmentation import main as instance_segmentation_main
from chatbot import main as chatbot_main


app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Models = {
    'image_captioning_model': None,
}

IMAGE_PROCESS_OK = 100
IMAGE_PROCESS_ERR = 101
INVALID_REQUEST_ERR = 231
INVALID_IMAGE_ERR = 232
UNKNOWN_ERR = 500

ERR_MESSAGES = {
    IMAGE_PROCESS_OK: 'The image is processed successfully.',
    IMAGE_PROCESS_ERR: 'The image process has been failed.',
    INVALID_REQUEST_ERR: 'Invalid request.',
    INVALID_IMAGE_ERR: 'Invalid image has input. Could not read the image data.',
    UNKNOWN_ERR: 'Unknown error has occurred.'
}


@app.route('/face-recognition/config-database', methods=['GET'])
def config_database():
    """
    Set the database connection
    """
    data = request.get_json()

    if 'host' not in data:
        return Response('"host" is missing.', status=400)

    port = 5432
    if 'port' not in data:
        port = data['port']
    
    if 'db_name' not in data:
        return Response('"db_name" is missing.', status=400)

    if 'username' not in data:
        return Response('"username" is missing.', status=400)
    
    if 'password' not in data:
        return Response('"password" is missing.', status=400)

    host = data['host']
    db_name = data['db_name']
    username = data['username']
    password = data['password']

    set_env('db_host', host)
    set_env('db_port', port)
    set_env('db_name', db_name)
    set_env('db_username', username)
    set_env('db_password', password)

    # Create tables for face recognition
    res = face_detection_main.set_db_engine()
    if not res:
        return Response('Database configuration is failed. Failed to create tables for face recognition module.', status=500)

    # Create tables for chatbot
    res = chatbot_main.create_chatbot()
    if not res:
        return Response('Database configuration is failed. Failed to create tables for chatbot module.', status=500)

    return Response('Database configuration is done.', status=200)


@app.route('/face-recognition/update-samples', methods=['GET', 'POST'])
def update_samples():
    # GET request
    if request.method == 'GET':
        return Response('Face detection server is running.', status=200)

    # POST request
    data_list = request.json

    ## Try to extract features from samples, and update database
    res_code, success_list, failure_list  = face_detection_main.update_sample_database(data_list)
    if res_code != face_detection_main.UPDATE_SAMPLE_FACES_OK:
        response = {
            'error': face_detection_main.ERR_MESSAGES[res_code]
        }
        return make_response(jsonify(response), 400)

    ## Make response
    response = {
        'success': [f['id'] for f in success_list],
        'fail': [f['id'] for f in failure_list]
    }
    return make_response(jsonify(response), 200)


@app.route('/face-recognition/clear-samples', methods=['GET', 'POST'])
def clear_samples():
    try:
        res = face_detection_main.clear_sample_database()

        if not res:
            response = {
                'error': 'Failed to clear the existing samples'
            }
            return make_response(jsonify(response), 500)

        response = {
            'success': 'Samples have been removed successfully'
        }
        
        return make_response(jsonify(response), 200)
    
    except Exception as e:
        response = {
            'fail': str(e)
        }
        return make_response(jsonify(response), 500)


@app.route('/face-recognition/update-metadata/<sample_id>', methods=['POST'])
def update_sample_metadata(sample_id):
    payloads = request.json
    if 'name' not in payloads:
        response = {
            'error': '"name" is missing in the request.'
        }
        return make_response(jsonify(response), 400)

    if 'metadata' not in payloads:
        response = {
            'error': '"metadata" is missing in the request.'
        }
        return make_response(jsonify(response), 400)

    config_data = {
        'name': payloads['name'],
        'metadata': payloads['metadata']
    }

    registered_samples = face_detection_main.get_sample_database()
    for r_sample in registered_samples:
        if r_sample['id'] == sample_id:
            break
    else:
        response = {
            'error': f'Could not find "{sample_id}" in database.'
        }
        return make_response(jsonify(response), 400)

    res, code = face_detection_main.update_unknown_sample_config(sample_id, config_data)
    if not res:
        if code == face_detection_main.DB_CONNECTION_ERR:
            response = {
                'error': 'DB connection error occurred'
            }
        else:
            response = {
                'error': 'Internal Server Error'
            }

        return make_response(jsonify(response), 500)

    response = {
        'success': f'"{sample_id}" has been configured successfully.'
    }
    return make_response(jsonify(response), 200)


@app.route('/face-recognition', methods=['GET', 'POST'])
def face_recognition():
    if request.method == 'GET':
        return Response('Face detection server is running.', status=200)

    # POST
    # Read image data
    img_data = request.json
    if 'image' not in img_data:
        response = {
            'error': face_detection_main.ERR_MESSAGES[face_detection_main.INVALID_REQUEST_ERR]
        }
        return make_response(jsonify(response), 400)

    # min_distance is optional parameter in request
    min_distance = 0.3  # default threshold for facenet, between 0 and 1
    if 'min_distance' in img_data:
        min_distance = float(img_data['min_distance'])

    # Process image
    start_time = datetime.now()
    res_code, candidates = face_detection_main.process_image(img_data['image'], min_distance)
    print(f'Image process takes {datetime.now() - start_time}')

    if res_code != face_detection_main.IMAGE_PROCESS_OK:
        response = {
            'error': face_detection_main.ERR_MESSAGES[res_code]
        }
        return make_response(jsonify(response), 500)

    # Return candidates
    return make_response(jsonify(candidates), 200)


@app.route('/image-captioning', methods=['GET', 'POST'])
def image_captioning_method():
    if request.method == 'GET':
        return Response('Image Captioning module is available.', status=200)

    # POST
    # Read image data
    img_data = request.json
    if 'image' not in img_data:
        response = {
            'error': ERR_MESSAGES[INVALID_REQUEST_ERR]
        }
        return make_response(jsonify(response), 400)

    if not face_detection_main.set_db_engine():
        response = {
            'error': 'DB connection error occurred'
        }
        return make_response(jsonify(response), 500)

    # Load model
    if Models['image_captioning_model'] is None:
        Models['image_captioning_model'] = image_captioning.load_model()

    # Process image
    caption = image_captioning.process_image(Models['image_captioning_model'], img_data['image'])

    # Chatterbot
    chatbot_question, chatbot_options = chatbot_main.run_chatterbot(caption)

    # Return candidates
    response = {
        'caption': caption,
        'question': chatbot_question,
        'options': chatbot_options
    }
    return make_response(jsonify(response), 200)


@app.route('/instance-segmentation/init-engine', methods=['GET'])
def instance_segmentation_init_engine():
    data = request.json

    if 'model' in data:
        print(f'Model: {data["model"]}')

        instance_segmentation_main.init_engine(data['model'])

        return Response('Loading model... Done.', status=200)
    
    return Response('Bad request. Your request is missing "model".', 400)


@app.route('/instance-segmentation/detect-objects', methods=['POST'])
def instance_segmentation_detect_objects():
    data = request.json

    if 'image' not in data:
        return Response('Bad request. Your request is missing "image".', 400)

    img_data = data['image']

    top_k = 15
    if 'top_k' in data:
        try:
            top_k = int(data['top_k'])
        except:
            pass

    score_threshold = 0.15
    if 'score_threshold' in data:
        try:
            score_threshold = float(data['score_threshold'])
        except:
            pass

    result = instance_segmentation_main.process_image(img_data, top_k, score_threshold)

    if result is None:
        return Response('Please init engine first by calling "/instance-segmentation/init-engine".', 500)

    return make_response(result, 200)