import requests

hostname = 'http://localhost:5000'


def get_generated_story():
    url = f'{hostname}/get-generated-story'

    response = requests.get(url=url)

    print(response.json())


if __name__ == '__main__':
    get_generated_story()