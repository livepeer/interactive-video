import io
import os
import json
import psycopg2
import requests
from dotenv import load_dotenv
from base64 import b64encode
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import (
    Base,
    SampleFaces,
    FeatureVectors,
    FaceImages
)

db_connection = None
db_session_factory = None

def get_env(key):
    try:
        #global env_path
        #load_dotenv(dotenv_path=env_path,override=True)
        load_dotenv(override=True)
        val = os.getenv(key)
        return val

    except :
        return None


def set_env(key, value):
    global env_path
    if key :
        if not value:
            value = '\'\''
        cmd = f'dotenv set -- {key} {value}'  # set env variable
        os.system(cmd)

def get_database_url(db_config):
    return f'postgresql+psycopg2://{db_config["username"]}:{db_config["password"]}@{db_config["host"]}:{db_config["port"]}/{db_config["db_name"]}'

def create_database(model_base, db_config):
    try:
        database_url = get_database_url(db_config)
        db_engine = create_engine(database_url)

        #if not db_engine.dialect.has_table(db_engine, 'sample_faces'):  # if there is no 'sample_faces' table, we will create tables.
        #    model_base.metadata.create_all(db_engine)

        return True

    except Exception as e:
        print(e)
        return False

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

def get_facelist():
    global db_session_factory
    if db_session_factory is None:
        res = set_db_engine()
        if not res:
            return None

    sample_vectors = []
    # Create an individual session
    db_session = db_session_factory()

    # Select all faces
    all_sample_faces = db_session.query(SampleFaces).all()

    for face in all_sample_faces:
        vectors = face.imgdatas        
        sample_vectors.append({
            'id': face.id,
            'sample_id': face.sample_id,
            'name': face.name,
            'metadata': face.meta_data,
            'action': face.action,            
            'imgdata': b64encode(vectors[0].imgdata).decode('utf-8')
        })

    db_session.close()

    return sample_vectors

def get_facedetail(id):
    global db_session_factory
    if db_session_factory is None:
        res = set_db_engine()
        if not res:
            return None
    # Create an individual session
    db_session = db_session_factory()
    sample_face = None
    face = db_session.query(SampleFaces).filter_by(id=id).first()
    
    vectors = face.imgdatas
    sample_face = {
            'id': face.id,
            'sample_id': face.sample_id,
            'name': face.name,
            'metadata': face.meta_data,
            'action': face.action,
            'imgdata': b64encode(vectors[0].imgdata).decode('utf-8')
    }
    db_session.close()

    return sample_face


def update_facedetail(id, sample_id, name, action, metadata):

    global db_session_factory
    if db_session_factory is None:
        res = set_db_engine()
        if not res:
            return None
    # Create an individual session
    db_session = db_session_factory()

    db_session.query(SampleFaces).filter_by(id=id).update({
            'sample_id': sample_id,
            'name': name,
            'action': action,
            'meta_data': metadata,
        })
    db_session.commit()

    # Close session
    db_session.close()


def delete_facedetail(id):
    global db_session_factory
    if db_session_factory is None:
        res = set_db_engine()
        if not res:
            return None
    # Create an individual session
    db_session = db_session_factory()

    db_session.query(SampleFaces).filter_by(id=id).delete()
    db_session.commit()

    # Close session
    db_session.close()

def set_db_connection():
    """
    Set Database Connection
    """
    global db_connection

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

    try:
        db_connection = psycopg2.connect(
            host=host,
            port=port,
            dbname=db_name,
            user=db_username,
            password=db_password
        )
    except Exception as e:
        print(e)
        return False

    return True

def get_facelist_q():
    global db_connection
    sample_vectors = []
    if db_connection is None:
        res = set_db_connection()
        if not res:
            return None
    cur = db_connection.cursor()
    cur.execute('SELECT id, created, modified, sample_id, name, metadata, action, imgdata FROM sample_face_vectors')
    sample_vector_list = cur.fetchall()


    for vector_data in sample_vector_list:
        vector = bytes(vector_data[7])
        sample_vectors.append({
            'id': vector_data[0],
            'sample_id': vector_data[3],
            'name': vector_data[4],
            'metadata': vector_data[5],
            'action': vector_data[6],
            'imgdata': b64encode(vector).decode('utf-8')
        })

    cur.close()
        
    return sample_vectors

def get_facedetail_q(id):
    global db_connection    
    if db_connection is None:
        res = set_db_connection()
        if not res:
            return None
    cur = db_connection.cursor()
    sample_face = None
    sql_query = f'SELECT id, created, modified, sample_id, name, metadata, action, imgdata FROM sample_face_vectors WHERE id = {id};'
    print(sql_query)
    cur.execute(sql_query)
    vector_data = cur.fetchall()
    if len(vector_data) > 0:
        vector = vector_data[0]
        sample_face = {
                'id': vector[0],
                'sample_id': vector[3],
                'name': vector[4],
                'metadata': vector[5],
                'action': vector[6],
                'imgdata': b64encode(vector[7]).decode('utf-8')
        }

    cur.close()
    return sample_face

def update_facedetail_q(id, sample_id, name, action, metadata):
    global db_connection
    if db_connection is None:
        res = set_db_connection()
        if not res:
            return

    cur = db_connection.cursor()
    sql_query = f'UPDATE sample_face_vectors SET sample_id = \'{sample_id}\', name =\'{name}\', action =\'{sample_id}\', metadata =\'{metadata}\' WHERE id = \'{id}\';'
    print(sql_query)
    cur.execute(sql_query)
    db_connection.commit()
    cur.close()

def delete_facedetail_q(id):
    global db_connection
    if db_connection is None:
        res = set_db_connection()
        if not res:
            return
    
    cur = db_connection.cursor()
    sql_query = f'DELETE FROM sample_face_vectors WHERE id = \'{id}\';'
    cur.execute(sql_query)
    db_connection.commit()
    cur.close()