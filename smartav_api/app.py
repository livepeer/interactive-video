import json
import requests
import threading
import torch
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, Response, jsonify, make_response, redirect, render_template

from face_recognition.face_detection import main as face_detection_main
from face_recognition.face_detection.common import set_env
from image_captioning import image_captioning
from instance_segmentation.segment import predict as instance_segmentation_predict
from chatbot import main as chatbot_main
from questgen import main


app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables
QGenerator = main.QGen()
Models = {
    'image_captioning_model': None,
}
last_image_captioning_list = []     # save the last image captioning results, length will be determined by call_story_gen_interval
call_story_gen_interval = 3
story_generator_is_free = True
generated_stories = []


# Constant
STORY_GENERATOR_API_HOST = 'http://10.23.11.105:8000'

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


# API content

@app.route('/api/docs')
def get_docs():
    print('sending docs')
    return render_template('swaggerui.html')


@app.route('/face-recognition/config-database', methods=['POST'])
def config_database():
    """
    Set the database connection
    """
    data = request.json

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


@app.route('/face-recognition/update-samples', methods=['POST'])
def update_samples():
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


@app.route('/face-recognition/clear-samples', methods=['POST'])
def clear_samples():
    try:
        res = face_detection_main.clear_sample_database()

        if not res:
            response = {
                'success': False,
                'msg': 'Failed to clear the existing samples'
            }
            return make_response(jsonify(response), 500)

        response = {
            'success': True,
            'msg': 'Samples have been removed successfully'
        }
        
        return make_response(jsonify(response), 200)
    
    except Exception as e:
        response = {
            'success': False,
            'error': str(e)
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


@app.route('/face-recognition', methods=['POST'])
def face_recognition():
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

# Begin Story Generator API
@app.route('/set-story-gen-api-host', methods=['POST'])
def set_story_gen_api_host():
    global STORY_GENERATOR_API_HOST

    params = request.json
    if 'hostname' not in params:
        response = {
            'error': ERR_MESSAGES[INVALID_REQUEST_ERR]
        }
        return make_response(jsonify(response), 400)

    STORY_GENERATOR_API_HOST = params['hostname']

    response = {
        'result': 'success'
    }
    return make_response(jsonify(response), 200)


@app.route('/set-story-gen-interval', methods=['POST'])
def set_call_story_generator_interval():
    global call_story_gen_interval

    params = request.json
    if 'interval' not in params:
        response = {
            'error': ERR_MESSAGES[INVALID_REQUEST_ERR]
        }
        return make_response(jsonify(response), 400)

    try:
        interval = int(params['interval'])
        call_story_gen_interval = interval
    except Exception:
        response = {
            'error': ERR_MESSAGES[INVALID_REQUEST_ERR]
        }
        return make_response(jsonify(response), 400)

    response = {
        'result': 'success'
    }
    return make_response(jsonify(response), 200)


@app.route('/init-story-generator', methods=['POST'])
def init_story_generator():
    """
    Initialize the story generator
    """
    url = f'{STORY_GENERATOR_API_HOST}/init-story-generator'
    return redirect(url, code=307)


@app.route('/get-generated-story', methods=['GET'])
def get_generated_story():
    global generated_stories
    
    response = {
        'results': generated_stories
    }
    return make_response(jsonify(response), 200)


@app.route('/add-text-to-story', methods=['POST'])
def add_text_to_story():
    """
    Initialize the story generator
    """
    url = f'{STORY_GENERATOR_API_HOST}/add-text-to-story'
    return redirect(url, code=307)
# End Story Generator API


def call_story_generator():
    global generated_stories, story_generator_is_free, last_image_captioning_list

    image_captioning_results_queue = last_image_captioning_list.copy()
    if len(image_captioning_results_queue) == 0:
        return

    story_generator_is_free = False

    # Call Story-Generator API
    url = f'{STORY_GENERATOR_API_HOST}/generate-story'
    headers = {'Content-Type': 'application/json'}
    data = {
        'prompt': '. '.join(image_captioning_results_queue)
    }
    res = requests.post(url, data=json.dumps(data), headers=headers)
    if res.status_code == 200:
        generated_stories = res.json()['results']

    story_generator_is_free = True


@app.route('/image-captioning', methods=['POST'])
def image_captioning_method():
    global last_image_captioning_list, story_generator_is_free

    # POST
    # Read image data
    img_data = request.json
    if 'image' not in img_data:
        response = {
            'error': ERR_MESSAGES[INVALID_REQUEST_ERR]
        }
        return make_response(jsonify(response), 400)

    candidate_size = 3
    if 'candidate_size' in img_data:
        candidate_size = img_data['candidate_size']

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

    # Trigger Story Generator
    last_image_captioning_list.append(caption)
    if len(last_image_captioning_list) >= call_story_gen_interval and story_generator_is_free:
        print('call story_generator', len(last_image_captioning_list))
        th = threading.Thread(target=call_story_generator)
        th.start()

        # AFter trigger the Story Generator, need to clear this list
        last_image_captioning_list = []

    # Chatterbot
    chatbot_result = chatbot_main.run_chatterbot(caption, candidate_size)

    # Return candidates
    response = {
        'caption': caption,
        'text': chatbot_result['text'],
        'questions': [{'id': quiz.id, 'question': quiz.text, 'options': json.loads(quiz.options)} for quiz in chatbot_result['candidates']]
    }
    
    return make_response(jsonify(response), 200)


@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    global QGenerator

    # Load speech content
    data = request.json
    if 'content' not in data:
        response = {
            'error': ERR_MESSAGES[INVALID_REQUEST_ERR]
        }
        return make_response(jsonify(response), 400)
    
    max_questions = 4
    if 'max_questions' in data:
        max_questions = int(data['max_questions'])

    content = data['content']

    payload = {
        'input_text': content,
        'max_questions': max_questions
    }
    output = QGenerator.predict_mcq(payload)

    return make_response(jsonify(output), 200)


@app.route('/instance-segmentation/load-model', methods=['POST'])
def instance_segmentation_load_model():
    data = request.json

    weights = 'instance_segmentation/yolov7-seg.pt'
    if 'weights' in data:
        weights = data['weights']

    device = 0
    if 'device' in data:
        device = data['device']
        if device not in [0, 1, 2, 3, 'cpu']:
            return Response('Bad request', status=400)

    dataset = 'instance_segmentation/data/coco.yml'
    if 'dataset' in data:
        dataset = data['dataset']
    
    half = False
    if 'half' in data:
        half = bool(data['half'])
    
    dnn = False
    if 'dnn' in data:
        dnn = bool(data['dnn'])
    
    res, error = instance_segmentation_predict.load_model(
        weights=weights,
        device=device,
        data=dataset,
        half=half,
        dnn=dnn
    )
    if res:
        return Response('success', status=200)
    
    return Response(error, status=500)


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

    iou_threshold = 0.45
    if 'iou_threshold' in data:
        try:
            iou_threshold = float(data['iou_threshold'])
        except:
            pass

    result = instance_segmentation_predict.process_image(
        image_data=img_data,
        weights='instance_segmentation/yolov7-seg.pt',
        conf_thres=score_threshold,
        iou_thres=iou_threshold,
        max_det=top_k,
    )

    return make_response(result, 200)