import json
import requests
from datetime import datetime

BASE_URL = 'http://127.0.0.1:8000/'


def test_generate_story():
    url = f'{BASE_URL}/generate-story'
    headers = {
        'Content-Type': 'application/json'
    }

    while True:
        prompt = input('> ').strip()
        if prompt == '/q':
            break

        payloads = {
            'action': prompt
        }
        start_date = datetime.now()
        res = requests.post(url, headers=headers, data=json.dumps(payloads))
        print(res.text)
        print(datetime.now() - start_date)

    print('Bye~~')


if __name__ == '__main__':
    test_generate_story()
