import base64
import json
import os
import requests
import time
from datetime import datetime

hostname = 'http://localhost:5000'


def config_database():
    url = f'{hostname}/face-recognition/config-database'

    headers = {
        'Content-Type': 'application/json'
    }

    payloads = {
        'host': '10.23.9.76',
        'port': '5432',
        'db_name': 'FaceRecognition',
        'username': 'admin',
        'password': 'admin123'
    }

    response = requests.get(url=url, headers=headers, json=payloads)

    print(response.text)


def update_sample_database_test():
    dir_path = os.path.join(os.path.dirname(__file__), 'face_recognition', 'images')
    valid_images = ['.jpg', '.gif', '.png']

    data_list = []
    for f in os.listdir(dir_path):
        filename = os.path.splitext(f)[0]
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        
        file_path = os.path.join(dir_path, f'{filename + ext}')
        with open(file_path, 'rb') as img_file:
            base64_img = base64.b64encode(img_file.read()).decode('utf-8')

            data_list.append({
                'id': 'james_test_image',
                'name': filename + ext,
                'image': f'data:image/{ext[1:]};base64,{base64_img}',
                'metadata': filename + ext,
                'action': 'embedlink'
            })
    
    url = f'{hostname}/face-recognition/update-samples'
    headers = {
        'Content-Type': 'application/json'
    }
    payload = json.dumps(data_list)

    response = requests.post(url, headers=headers, data=payload)

    print(response.text)


def clear_samples():
    url = f'{hostname}/face-recognition/clear-samples'
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.get(url, headers=headers, data={})

    print(response.text)


def update_sample_metadata(sample_id):
    url = f'{hostname}/face-recognition/update-metadata/{sample_id}'
    headers = {
        'Content-Type': 'application/json'
    }
    payloads = {
        'name': 'James',
        'metadata': 'James Metadata'
    }
    response = requests.post(url, headers=headers, data=json.dumps(payloads))

    print(response.text)


def face_recognition_test(number):
    dir_path = os.path.join(os.path.dirname(__file__), 'face_recognition', 'images')
    file_path = os.path.join(dir_path, f'img{number}.jpg')
    with open(file_path, 'rb') as img_file:
        base64_img = base64.b64encode(img_file.read()).decode('utf-8')

        img_data = {
            'image': f'data:image/jpeg;base64,{base64_img}',
            'min_distance': 0.3
        }

    url = f'{hostname}/face-recognition'
    headers = {
        'Content-Type': 'application/json'
    }
    payload = json.dumps(img_data)

    response = requests.post(url, headers=headers, data=payload)

    print(response.text)


def image_captioning_test():
    dir_path = os.path.join(os.path.dirname(__file__), 'image_captioning', 'images')
    file_path = os.path.join(dir_path, f'test_image_6.jpg')
    with open(file_path, 'rb') as img_file:
        base64_img = base64.b64encode(img_file.read()).decode('utf-8')

        img_data = {
            'image': f'data:image/jpeg;base64,{base64_img}',
        }

    url = f'{hostname}/image-captioning'
    headers = {
        'Content-Type': 'application/json'
    }
    payload = json.dumps(img_data)

    response = requests.post(url, headers=headers, data=payload)

    print(response.text)


def init_story_generator_test():
    url = f'{hostname}/init-story-generator'

    headers = {
        'Content-Type': 'application/json'
    }

    prompt = input('> Initialize the Story: ')
    payloads = {
        'prompt': prompt
    }

    res = requests.post(url, headers=headers, data=json.dumps(payloads))

    print(res.json())


def story_generator_test():
    dir_path = os.path.join(os.path.dirname(__file__), 'image_captioning', 'images')

    url = f'{hostname}/image-captioning'
    headers = {
        'Content-Type': 'application/json'
    }

    for i in range(100):
        filename = f'test_image_{i % 6 + 1}.jpg'
        file_path = os.path.join(dir_path, filename)

        with open(file_path, 'rb') as img_file:
            base64_img = base64.b64encode(img_file.read()).decode('utf-8')

            img_data = {
                'image': f'data:image/jpeg;base64,{base64_img}',
            }

        payload = json.dumps(img_data)

        response = requests.post(url, headers=headers, data=payload)

        print(response.text)

        time.sleep(20)


def test_add_text_to_story():
    url = f'{hostname}/add-text-to-story'

    headers = {
        'Content-Type': 'application/json'
    }

    text = input('> Input the text that should be added to story: ')
    payloads = {
        'text': text
    }

    res = requests.post(url, headers=headers, data=json.dumps(payloads))

    print(res.json())


def test_generate_questions():
    url = f'{hostname}/generate-questions'
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        'content': 'Sachin Ramesh Tendulkar is a former international cricketer from India and a former captain of the Indian national team. He is widely regarded as one of the greatest batsmen in the history of cricket. He is the highest run scorer of all time in International cricket.',
        'max_questions': 5
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(response.status_code, response.text)


def instance_segmentation_load_model_test():
    url = f'{hostname}/instance-segmentation/load-model'

    headers = {
        'Content-Type': 'application/json'
    }

    payloads = {
        "weights": "instance_segmentation/yolov7-seg.pt"
    }

    res = requests.post(url, headers=headers, data=json.dumps(payloads))

    print(res.status_code)
    print(res.text)


def instance_segmentation_detect_objects_test():
    dir_path = os.path.join(os.path.dirname(__file__), 'instance_segmentation', 'inputs')
    file_path = os.path.join(dir_path, f'james_test_image_1.jpg')
    with open(file_path, 'rb') as img_file:
        base64_img = base64.b64encode(img_file.read()).decode('utf-8')

        payload = {
            'image': f'data:image/jpeg;base64,{base64_img}',
            'top_k': 12,
            'score_threshold': 0.4
        }

    url = f'{hostname}/instance-segmentation/detect-objects'
    headers = {
        'Content-Type': 'application/json'
    }

    res = requests.post(url, headers=headers, data=json.dumps(payload))

    print(res.status_code)
    print(res.text)


if __name__ == '__main__':
    # # # Configure Database
    # config_database()

    # # # Clear samples
    # clear_samples()

    # # # Update sample database with the images in /dataset directory
    # update_sample_database_test()

    # # Update sample metadata
    # update_sample_metadata('unknown_person_1')

    # # Test question generator
    # test_generate_questions()

    # for i in range(0, 10000):
        # # # Test face recognition with "img1.jpg"
        # face_recognition_test(1)

        # # Test image captioning with "test_image_1.jpg"
        # image_captioning_test()

        # # Test instance segmentation init engine
        # instance_segmentation_load_model_test()

        # # Test instance segmentation detect objects
        # instance_segmentation_detect_objects_test()

    # init_story_generator_test()
    
    story_generator_test()

    # test_add_text_to_story()
