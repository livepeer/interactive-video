from flask import Flask, request, Response, jsonify, make_response

from generator.gpt2.gpt2_generator import *
from story.story_manager import *
from story.utils import first_to_second_person

app = Flask(__name__)
story_manager = None


def process_action(action):
    """
    Process the user-input to determine it's for the first/second person
    """
    action = action.strip()

    if "you" not in action[:6].lower() and "I" not in action[:6]:
        action = action[0].lower() + action[1:]
        action = "You " + action

    if action[-1] not in [".", "?", "!"]:
        action = action + "."

    action = first_to_second_person(action)

    return action


@app.route('/init-story-generator', methods=['POST'])
def init_story_generator():
    """
    Initialize the story generator
    """
    params = request.json
    if 'prompt' not in params:
        response = {
            'error': 'Invalid request. The request should contain "prompt".'
        }
        return make_response(jsonify(response), 400)

    prompt = params['prompt']

    # Initialize story_manager
    global story_manager

    if story_manager is None:
        generator = GPT2Generator()
        story_manager = UnconstrainedStoryManager(generator)
        
    result = story_manager.start_new_story(prompt)

    # Return response
    response = {
        'result': result
    }

    return make_response(jsonify(response), 200)


@app.route('/generate-story', methods=['POST'])
def generate_story():
    """
    Generate Story
    Request should contain the prompt
    """
    # Get params from request
    params = request.json

    if 'prompt' not in params:
        response = {
            'error': 'Invalid request. The request should contain "prompt".'
        }
        return make_response(jsonify(response), 400)

    prompt = params['prompt']

    global story_manager
    if story_manager is None:
        response = {
            'error': 'Please initialize the story generator first.'
        }
        return make_response(jsonify(response), 500)

    # Process input action
    print(prompt)
    action = process_action(action=prompt)
    action = "\n> " + action + "\n"

    # Generate the next story
    results = story_manager.act(action, result_type='multiple')

    # Return candidates
    response = {
        'results': results
    }

    return make_response(jsonify(response), 200)


@app.route('/add-text-to-story', methods=['POST'])
def add_text_to_story():
    """
    Generate Story
    Request should contain the prompt
    """
    # Get params from request
    params = request.json

    if 'text' not in params:
        response = {
            'error': 'Invalid request. The request should contain "text".'
        }
        return make_response(jsonify(response), 400)

    text = params['text']

    global story_manager
    if story_manager is None:
        response = {
            'error': 'You didn\'t initialize the story generator.'
        }
        return make_response(jsonify(response), 500)

    res = story_manager.add_text(text)
    if res:
        return make_response(jsonify({'result': 'success'}), 200)

    return make_response(jsonify({'result': 'failure'}), 500)
