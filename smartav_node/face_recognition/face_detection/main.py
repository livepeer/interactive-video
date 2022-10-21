import io
import os
import json
import numpy as np
import requests
import threading
from datetime import datetime
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from PIL import Image as im
from requests import RequestException, ConnectionError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .load_image import load_image
from .models import (
    Base,
    SampleFaces,
    FeatureVectors
)
from .common import (
    get_env,
    set_env,
    get_database_url,
    create_database
)


FEATURE_EXTRACTION_URL = 'http://127.0.0.1:8000/extract_feature'
DIR_PATH = "dataset/"
SAMPLE_FACE_VECTOR_DATABASE = []
FEATURE_EXTRACT_BATCH_SIZE = 10

IMAGE_PROCESS_OK = 100
IMAGE_PROCESS_ERR = 101
EXTRACT_SAMPLE_VECTOR_OK = 200
EXTRACT_SAMPLE_VECTOR_ERR = 201
UPDATE_SAMPLE_FACES_OK = 202
UPDATE_SAMPLE_FACES_ERR = 203
FEATURE_EXTRACTION_SERVER_CONNECTION_ERR = 204
FEATURE_EXTRACTION_REQUEST_ERR = 205
FEATURE_EXTRACTION_SERVER_RESPONSE_OK = 206
FEATURE_EXTRACTION_SERVER_RESPONSE_PARSE_ERR = 207
DB_CONNECTION_ERR = 208
FACE_DETECTION_OK = 210
FACE_DETECTION_ERR = 211
NO_FACE_DETECTED_ERR = 212
CALC_DISTANCE_OK = 220
CALC_DISTANCE_ERR = 221
NO_SAMPLE_VECTOR_ERR = 222
GET_SAMPLE_VECTOR_ERR = 223
NO_SUCH_FILE_ERR = 230
INVALID_REQUEST_ERR = 231
INVALID_IMAGE_ERR = 232
UNKNOWN_ERR = 500

ERR_MESSAGES = {
    IMAGE_PROCESS_OK: 'The image is processed successfully.',
    IMAGE_PROCESS_ERR: 'The image process has been failed.',
    UPDATE_SAMPLE_FACES_OK: 'Sample vector database has been updated successfully.',
    UPDATE_SAMPLE_FACES_ERR: 'Failed to update the sample vector database.',
    FEATURE_EXTRACTION_SERVER_CONNECTION_ERR: 'Feature extraction node is not running.',
    FEATURE_EXTRACTION_REQUEST_ERR: 'Bad request to feature extraction node.',
    FEATURE_EXTRACTION_SERVER_RESPONSE_OK: 'Successfully received a response from feature extraction node.',
    FEATURE_EXTRACTION_SERVER_RESPONSE_PARSE_ERR: 'Failed to parse a response from feature extraction node.',
    DB_CONNECTION_ERR: 'Database Connection Error',
    FACE_DETECTION_OK: 'Faces are successfully detected from the input image.',
    FACE_DETECTION_ERR: 'Failed to detect face from the input image',
    NO_FACE_DETECTED_ERR: 'No face detected from the input image.',
    CALC_DISTANCE_OK: 'Calculation of vector distance has been suceeded.',
    CALC_DISTANCE_ERR: 'Failed to calculate the vector distance.',
    NO_SAMPLE_VECTOR_ERR: 'There is no sample face data.',
    GET_SAMPLE_VECTOR_ERR: 'Couldn\'t get sample data, it seems like there is no database connection.',
    NO_SUCH_FILE_ERR: 'No such file.',
    INVALID_REQUEST_ERR: 'Invalid request.',
    INVALID_IMAGE_ERR: 'Invalid image has input. Could not read the image data.',
    UNKNOWN_ERR: 'Unknown error has occurred.'
}

face_detection_model = None
db_session_factory = None


def load_model():
    """
    Load Model
    """
    global face_detection_model

    # use FaceAnalysis
    face_detection_model = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection'])
    face_detection_model.prepare(ctx_id=0, det_size=(160, 160))


def set_db_engine():
    """
    Set Database Connection
    """
    global db_session_factory

    load_dotenv(override=True)
    host = get_env('db_host')
    port = get_env('db_port')
    db_name = get_env('db_name')
    db_username = get_env('db_username')
    db_password = get_env('db_password')

    if host is None or db_name is None or db_username is None or db_password is None:
        return False

    if port is None:
        port = 5432     # 5432 as default port of postgres

    db_config = {
        'host': host,
        'port': port,
        'db_name': db_name,
        'username': db_username,
        'password': db_password
    }

    try:
        # Database URI
        db_url = get_database_url(db_config)

        # Create tables if there is no table
        res = create_database(
            model_base=Base,
            db_config=db_config
        )

        if not res:
            return False
        
        # Create an engine
        db_engine = create_engine(db_url, convert_unicode=True)

        # Publish a database session
        db_session_factory = sessionmaker(bind=db_engine)

    except Exception as e:
        print(e)
        return False

    return True


def detect_faces(img):
    """
    Detect the faces from the base image
    """
    global face_detection_model

    if face_detection_model is None:
        load_model()

    if isinstance(img, str):
        if len(img) == 0:
            return None, None
        elif len(img) > 11 and img[0:11] == "data:image/":
            img = load_image(img)
        else:
            img = load_image(DIR_PATH + img)

    elif isinstance(img, bytes):
        img = np.array(im.open(io.BytesIO(img)))

    else:
        return None, None

    if not isinstance(img, np.ndarray):
        return None, None

    # faces stores list of bbox and kps
    faces = face_detection_model.get(img)  # [{'bbox': [], 'kps': []}, {'bbox': [], 'kps': []}, ...]

    # no face is detected
    if len(faces) == 0:
        print("No face detected")
        return img, None

    return img, faces


def call_feature_extractor(face_list):
    """
    Send request to feature extraction node. Request will contain list of face ids and detected face image
    Returns error code, and result string
    """
    success_feature_vectors = []
    failure_feature_vectors = []

    try:
        start_time = datetime.now()
        face_data = []
        image_files = []
        for f in face_list:
            face_data.append({
                'id': f['id']
            })

            # Convert the numpy array to bytes
            face_pil_img = im.fromarray(f['img'])
            byte_io = io.BytesIO()
            face_pil_img.save(byte_io, 'png')
            byte_io.seek(0)

            image_files.append((
                'image', byte_io
            ))

        # Send request to feature extraction node
        response = requests.post(FEATURE_EXTRACTION_URL, data={'face_data': json.dumps(face_data)}, files=image_files)

        # Parse the response and get the feature vectors
        try:
            feature_list = json.loads(response.text)

            # Determine which one is success, which one is failure
            for fe in feature_list:
                if len(fe['vector']) == 0:  # If feature extraction is failed
                    failure_feature_vectors.append({
                        'id': fe['id']
                    })
                else:   # If feature extraction is suceed
                    success_feature_vectors.append({
                        'id': fe['id'],
                        'vector': np.array(fe['vector'])
                    })

        except:
            return FEATURE_EXTRACTION_SERVER_RESPONSE_PARSE_ERR, None, None

    except ConnectionError:
        return FEATURE_EXTRACTION_SERVER_CONNECTION_ERR, None, None

    except RequestException:
        return FEATURE_EXTRACTION_REQUEST_ERR, None, None

    return FEATURE_EXTRACTION_SERVER_RESPONSE_OK, success_feature_vectors, failure_feature_vectors


def feature_extraction_thread(face_list, extract_success_list, extract_failure_list):
    """
    Call feature extraction module. this function will be run in multi-threads
    """
    # Prepare the MetaData's Map, this will be useful to determine which faces are success to extract features, and failure
    metadata_map = {}
    for f in face_list:
        metadata_map[f['id']] = f

    # Call API of feature extraction server
    res_code, success_face_features, failure_face_features = call_feature_extractor(face_list)
            
    if res_code != FEATURE_EXTRACTION_SERVER_RESPONSE_OK:
        # Add all faces to failed list
        for f in face_list:
            extract_failure_list.append({
                'id': f['id']
            })
        return

    # Treat the success faces, add meta data
    for face in success_face_features:
        # If could not find meta data of this face, move it to failed list
        if face['id'] not in metadata_map:
            failure_face_features.append(face)
            continue

        # Add meta data
        meta_data = metadata_map[face['id']]
        face['name'] = meta_data['name']
        face['metadata'] = meta_data['metadata']
        face['action'] = meta_data['action']

    # Append to result arrays
    extract_success_list += success_face_features
    extract_failure_list += failure_face_features


def extract_sample_feature_vector(data_list):
    """
    Extract the feature vector from the sample images
    Return code, extract_success_list, extract_failure_list
    """
    face_list = []
    extract_success_list = []
    extract_failure_list = []
    thread_pool = []

    # Main loop, each element will contain one image and its metadata
    for data in data_list:
        try:
            sample_id = data['id']
            name = data['name']
            img = data['image']
            metadata = data['metadata']
            action = data['action']
        except:
            return INVALID_REQUEST_ERR, None, None

        # Detect face from sample image
        base_img, detected_faces = detect_faces(img)

        # No face detected
        if detected_faces is None:
            continue

        # Get the first face from the detected faces list. Suppose that the sample image has only 1 face
        face = detected_faces[0]    # {'bbox': [x1, y1, x2, y2], 'kps': []}

        # # Get face region from the base image(profile image)
        bbox = face['bbox']
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        face_img = base_img[y1:y2,x1:x2]

        face_list.append({
            'id': sample_id,
            'img': face_img,
            'name': name,
            'metadata': metadata,
            'action': action
        })

        if len(face_list) == FEATURE_EXTRACT_BATCH_SIZE:
            th = threading.Thread(target=feature_extraction_thread, args=(face_list, extract_success_list, extract_failure_list))
            th.start()
            thread_pool.append(th)

            face_list = []

    if len(face_list) > 0:
        th = threading.Thread(target=feature_extraction_thread, args=(face_list, extract_success_list, extract_failure_list))
        th.start()
        thread_pool.append(th)

    # Wait until all threads are finished
    for th in thread_pool:
        th.join()

    return EXTRACT_SAMPLE_VECTOR_OK, extract_success_list, extract_failure_list


def save_sample_database(sample_vectors):
    """
    Save the sample face feature vector into the database
    """
    global db_session_factory
    if db_session_factory is None:
        res = set_db_engine()
        if not res:
            return DB_CONNECTION_ERR

    # Create an individual session
    db_session = db_session_factory()

    # Run query
    for vector_data in sample_vectors:
        sample_id = vector_data['id']
        name = vector_data['name']
        metadata = vector_data['metadata']
        action = vector_data['action']
        vector = vector_data['vector']

        if db_session.query(SampleFaces).filter_by(sample_id=sample_id).count() > 0:
            # Append a new feature vector to existing face
            face = db_session.query(SampleFaces).filter_by(sample_id=sample_id).first()
            face.vectors.append(
                FeatureVectors(
                    vector=json.dumps(vector.tolist())
                )
            )

        else:
            # Save new face
            new_face = SampleFaces(
                sample_id=sample_id,
                name=name,
                meta_data=metadata,
                action=action
            )
            new_face.vectors.append(
                FeatureVectors(
                    vector=json.dumps(vector.tolist())
                )
            )
            db_session.add(new_face)

    db_session.commit()
    db_session.close()

    return UPDATE_SAMPLE_FACES_OK


def get_sample_database():
    """
    Read sample feature vector from database
    """
    global db_session_factory
    sample_vectors = []
    
    if db_session_factory is None:
        res = set_db_engine()
        if not res:
            return None

    # Create an individual session
    db_session = db_session_factory()

    # Select all faces
    all_sample_faces = db_session.query(SampleFaces).all()

    for face in all_sample_faces:
        vectors = face.vectors
        for v in vectors:
            fv = np.array(json.loads(v.vector))

            sample_vectors.append({
                'id': face.sample_id,
                'name': face.name,
                'metadata': face.meta_data,
                'action': face.action,
                'vector': fv
            })

    db_session.close()

    return sample_vectors


def clear_sample_database():
    """
    Clear all faces and vectors
    """
    global db_session_factory
    
    try:
        if db_session_factory is None:
            res = set_db_engine()
            if not res:
                return None
        
        # Create an individual session
        db_session = db_session_factory()

        # Delete all faces
        db_session.query(SampleFaces).delete()

        # Delete all vectors
        db_session.query(FeatureVectors).delete()

        db_session.commit()
        db_session.close()

    except Exception as e:
        print(e)
        return False

    return True


def register_unknown_face(face_vector):
    """
    This function will register a face_vector as an unknown person's face into database.
    """
    try:
        # Get database connection
        global db_session_factory

        if db_session_factory is None:
            res = set_db_engine()
            if not res:
                return DB_CONNECTION_ERR

        # Create an individual session
        db_session = db_session_factory()

        # Prepare fields
        ## Check the already registered unknown faces
        unknown_faces = db_session.query(SampleFaces).filter(SampleFaces.sample_id.ilike('unknown_person_'))

        ## Calculate the suffix_id for unknown face
        suffix_id = 0
        for uf in unknown_faces:
            sample_id = uf.sample_id
            try:
                last_suffix_id = int(sample_id.split('unknown_person_')[1])
                if last_suffix_id >= suffix_id:
                    suffix_id = last_suffix_id
            except Exception as e:
                print(str(e))
                continue
        suffix_id += 1

        sample_id = f'unknown_person_{suffix_id}'
        name = sample_id
        metadata = sample_id
        action = 'embedlink'

        # Save new vector
        new_face = SampleFaces(
            sample_id=sample_id,
            name=name,
            meta_data=metadata,
            action=action
        )
        new_face.vectors.append(
            FeatureVectors(
                vector=json.dumps(face_vector.tolist())
            )
        )
        db_session.add(new_face)

        db_session.commit()
        db_session.close()

        sample_face = {
            'id': sample_id,
            'name': name,
            'metadata': metadata,
            'action': action,
            'vector': face_vector
        }

        return True, sample_face

    except Exception as e:
        print(e)
        return False, None


def update_unknown_sample_config(sample_id, config_data):
    try:
        global db_session_factory

        if db_session_factory is None:
            res = set_db_engine()
            if not res:
                return False, DB_CONNECTION_ERR

        # Create an individual session
        db_session = db_session_factory()

        # Update
        db_session.query(SampleFaces).filter_by(sample_id=sample_id).update({
            'name': config_data['name'],
            'meta_data': config_data['metadata']
        })
        db_session.commit()

        # Close session
        db_session.close()

        return True, UPDATE_SAMPLE_FACES_OK

    except Exception as e:
        return False, UPDATE_SAMPLE_FACES_ERR


def calculate_simulation(feat1, feat2):
    from numpy.linalg import norm
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim


def find_face(face_feature_vectors, min_simulation):
    """
    Find the closest sample by comparing the feature vectors
    """
    # Read sample database
    sample_vectors = get_sample_database()

    if sample_vectors is None:
        return GET_SAMPLE_VECTOR_ERR, None

    candidates = []
    for vector_data in face_feature_vectors:
        face_feature_vector = vector_data['vector']

        # Initialize variables
        closest_id = ''
        closest_name = ''
        closest_metadata = ''
        closest_simulation = -1
        
        # Compare with sample vectors
        for i in range(len(sample_vectors)):
            sample = sample_vectors[i]
            sample_vector = sample['vector']

            try:
                # Calculate the distance between sample and the detected face.
                simulation = calculate_simulation(face_feature_vector, sample_vector)
            
                if (closest_id == '' or simulation > closest_simulation) and simulation > min_simulation:
                    closest_simulation = simulation
                    closest_id = sample['id']
                    closest_name = sample['name']
                    closest_metadata = sample['metadata']

            except Exception as e:
                pass
        
        # If not find fit sample, register this face as unkown person in database
        if closest_id == '':
            res, face_obj = register_unknown_face(face_feature_vector)
            if res:
                sample_vectors.append(face_obj)

                # Add candidate for this face
                candidates.append({
                    'id': face_obj['id'],
                    'name': face_obj['name'],
                    'metadata': face_obj['metadata'],
                    'bbox': vector_data['bbox']
                })
            continue

        # Add candidate for this face
        candidates.append({
            'id': closest_id,
            'name': closest_name,
            'metadata': closest_metadata,
            'bbox': vector_data['bbox']
        })
        
    return CALC_DISTANCE_OK, candidates


def process_image(img, min_distance):
    """
    Face recognition
    """
    # Detect the faces from the image that is dedicated in the path or bytes
    try:
        base_img, faces = detect_faces(img)
    except:
        return FACE_DETECTION_ERR, None

    if len(faces) == 0:
        return NO_FACE_DETECTED_ERR, None

    bound_box_map = {}
    face_list = []
    face_feature_vector_list = []

    # Send request to feature_extraction module
    for i in range(len(faces)):
        face = faces[i]     # [{'bbox': [], 'kps': []}, {'bbox': [], 'kps': []}, ...]

        bbox = face['bbox']
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        w = x2 - x1
        h = y2 - y1
        face_img = base_img[y1:y2,x1:x2]

        # Prepare bound box map
        bound_box_map[i] = f'{x1}, {y1}, {w}, {h}'

        # Make the face list, I will send bunch of faces to Feature Extraction Server at once
        face_list.append({
            'id': i,
            'img': face_img
        })

        if len(face_list) == FEATURE_EXTRACT_BATCH_SIZE:
            # Call the api to extract the feature from the detected faces
            res_code, success_face_features, failure_face_features = call_feature_extractor(face_list)

            if res_code != FEATURE_EXTRACTION_SERVER_RESPONSE_OK:
                return res_code, None

            face_feature_vector_list += success_face_features

            face_list = []

    if len(face_list) > 0:
        # Call the api to extract the feature from the detected faces
        res_code, success_face_features, failure_face_features = call_feature_extractor(face_list)

        if res_code != FEATURE_EXTRACTION_SERVER_RESPONSE_OK:
            return res_code, None

        face_feature_vector_list += success_face_features
    
    # Add bound box for each face feature vector
    vector_list = []
    for f in face_feature_vector_list:
        if int(f['id']) not in bound_box_map:
            continue
        
        f['bbox'] = bound_box_map[int(f['id'])]
        vector_list.append(f)

    # Find candidates by comparing feature vectors between detected face and samples
    status, candidates = find_face(vector_list, min_distance)

    if status != CALC_DISTANCE_OK:
        return status, None

    return IMAGE_PROCESS_OK, candidates


def update_sample_database(data_list):
    """
    Update the database that contains sample face vectors
    """
    # Extract the feature vector
    res, success_sample_vectors, failure_sample_vectors = extract_sample_feature_vector(data_list)

    if res != EXTRACT_SAMPLE_VECTOR_OK:
        return res, success_sample_vectors, failure_sample_vectors

    # Save the sample feature vector into database
    res = save_sample_database(success_sample_vectors)

    return res, success_sample_vectors, failure_sample_vectors
