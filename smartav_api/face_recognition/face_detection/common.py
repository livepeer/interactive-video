import os
from dotenv import load_dotenv
from sqlalchemy import create_engine


def get_env(key):
    try:
        #global env_path
        #load_dotenv(dotenv_path=env_path,override=True)
        load_dotenv(override=True)
        val = os.getenv(key)
        return val

    except:
        return None

def set_env(key, value):
    global env_path
    if key :
        if not value:
            value = '\'\''
        cmd = f'dotenv set -- {key} {value}'  # set env variable
        os.system(cmd)

def get_database_url(db_config):
    return f'postgres+psycopg2://{db_config["username"]}:{db_config["password"]}@{db_config["host"]}:{db_config["port"]}/{db_config["db_name"]}'

def make_database_url_from_env():
    load_dotenv(override=True)
    host = get_env('db_host')
    port = get_env('db_port')
    db_name = get_env('db_name')
    db_username = get_env('db_username')
    db_password = get_env('db_password')

    if host is None or db_name is None or db_username is None or db_password is None:
        return False
    
    return f'postgres+psycopg2://{db_username}:{db_password}@{host}:{port}/{db_name}'


def create_database(model_base, db_config):
    try:
        database_url = get_database_url(db_config)
        db_engine = create_engine(database_url)

        if not db_engine.dialect.has_table(db_engine, 'sample_faces'):  # if there is no 'sample_faces' table, we will create tables.
            model_base.metadata.create_all(db_engine)

        return True

    except Exception as e:
        print(e)
        return False
