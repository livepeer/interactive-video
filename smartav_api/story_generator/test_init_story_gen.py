import json
import requests

BASE_URL = 'http://127.0.0.1:8000/'


def init_story_generator():
    url = f'{BASE_URL}/init-story-generator'

    headers = {
        'Content-Type': 'application/json'
    }

    prompt = input('> Initialize the Story: ')
    payloads = {
        'prompt': prompt
    }

    res = requests.post(url, headers=headers, data=json.dumps(payloads))

    print(res.status_code)
    print(res.text)

if __name__ == '__main__':
    init_story_generator()
