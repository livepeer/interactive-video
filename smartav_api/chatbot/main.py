import json
import os
from chatbot.chatterbot.chatterbot import ChatBot
from chatbot.chatterbot.trainers import ChatterBotCorpusTrainer

chatbot = None
trainer = None


def create_chatbot():
    global chatbot, trainer

    if chatbot:
        return True

    try:
        chatbot = ChatBot('James', maximum_similarity_threshold=0.7)

        # Create a new trainer for the chatbot
        trainer = ChatterBotCorpusTrainer(chatbot)

        # Train the chatbot based on the english corpus
        dir_path = os.path.join(os.path.dirname(__file__), 'chatterbot-corpus', 'data')
        corpus_file_path = os.path.join(dir_path, f'corpus.yml')
        trainer.train(corpus_file_path)

    except Exception as e:
        print(e)
        return False

    return True
    

def run_chatterbot(input_text):
    global chatbot

    if chatbot is None:
        create_chatbot()

    res = chatbot.get_response(input_text)
    text = res.text
    options = json.loads(res.options)

    print('\n====================')
    print(f'Input: {input_text}')
    print('--------------------')
    print(text)

    print('\nOptions: ')

    for i in range(0, len(options)):
        print(f'{i+1}. {options[i]}')

    print('====================\n\n')

    return text, options
