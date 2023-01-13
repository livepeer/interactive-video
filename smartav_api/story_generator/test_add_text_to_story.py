import json
import requests

BASE_URL = 'http://127.0.0.1:8000/'


def add_text_to_story():
    url = f'{BASE_URL}/add-text-to-story'

    headers = {
        'Content-Type': 'application/json'
    }

    text = input('> Input the text that should be added to story: ')
    payloads = {
        'text': text
    }

    res = requests.post(url, headers=headers, data=json.dumps(payloads))

    print(res.status_code)
    print(res.text)

if __name__ == '__main__':
    add_text_to_story()
