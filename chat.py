import random
import json
import pyttsx3
import torch
from pydub import AudioSegment
TOKEN = "1557351267:AAF45Cr3QU9dNpshdv7R7KgiGmgvfOOZJbY"
# import pydub
# pydub.AudioSegment.ffmpeg = "/absolute/path/to/ffmpeg"p
import speech_recognition as sr
what_did_you_say = sr.Recognizer()
import requests

from bs4 import BeautifulSoup
import googlesearch
import wikipedia

def wiki(query):
    try:
        p = wikipedia.page(query)
    except:
        return "you need to be more specific."
    content = p.content # Content of page.
    return content.split(".")[0]+"."
def get_text_from_url(url):
    res = requests.get(url)
    html_page = res.content
    soup = BeautifulSoup(html_page, 'html.parser')
    text = soup.find_all(text=True)
    
    output = ''
    blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head', 
        'input',
        'script',
        # there may be more elements you don't want, such as "style", etc.
    ]
    
    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t)
    
    return output.split(".")[2]

def first_googleresult(query):
    return googlesearch.search(query)[0]

url = "https://api.telegram.org/bot" + TOKEN + "/"
##########################TRYING TO SEND AUDIO
import telegram
bot = telegram.Bot(token=TOKEN)
##############################################

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
    
def send_mess(chat, text):
    chat = last_update(get_updates_json(url))['message']['from']['id']
    params = {'chat_id': chat, 'text': text}
    response = requests.post(url + 'sendMessage', data=params)
    return response

def get_voice_message():
    path = last_update(get_updates_json(url))["message"]['voice']['file_id']
    # ID = last_update(get_updates_json(url))["message"]["message_id"]
    voice_path = "https://api.telegram.org/bot" + TOKEN + "/getFile?file_id="+path
    path = requests.get(voice_path).json()["result"]["file_path"]
    download_link  = "https://api.telegram.org/file/bot" + TOKEN + "/"+path
    r = requests.get(download_link)
    with open("voice.ogg",'wb') as f: 
        f.write(r.content)
    sound = AudioSegment.from_ogg("voice.ogg")
    sound.export("voice.wav", format="wav")
def get_updates_json(request):
    response = requests.get(request + 'getUpdates')
    return response.json()


def last_update(data):
    results = data['result']
    total_updates = len(results) - 1
    return results[total_updates]


FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")

mssg_ID = last_update(get_updates_json(url))['message']['message_id']
while(True):
        if(mssg_ID != last_update(get_updates_json(url))['message']['message_id']):
            break    
while True:
    
    try:
        sentence = last_update(get_updates_json(url))['message']['text']
    except:
        get_voice_message()
        with sr.AudioFile("voice.wav") as source:
            audio = what_did_you_say.record(source)
        try:
            s = what_did_you_say.recognize_google(audio)
            sentence = s
        except:
            sentence = "OOGA BOOGA"
    mssg_ID = last_update(get_updates_json(url))['message']['message_id']
    if sentence == "quit":
        break
    untokenized = sentence
    print(untokenized)
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.80:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                engine = pyttsx3.init()
                # engine.say({random.choice(intent['responses'])})
                answered = random.choice(intent['responses'])
                engine.save_to_file({answered}, './audio.ogg')
                engine.runAndWait()
                bot.send_audio(chat_id=last_update(get_updates_json(url))['message']['from']['id'], audio=open('./audio.ogg', 'rb'))
                send_mess(1, {answered})
    else:
        # engine = pyttsx3.init()
        # engine.say("Sorry, I don't understand.")
        send_mess(1, str("I think " + wiki(untokenized)))
        googleRes = "Also, try visiting - " + first_googleresult(untokenized)
        send_mess(1, googleRes)
        #engine.save_to_file('Sorry. I dont understand.', './audio.mp3')
        # engine.runAndWait()
        # bot.send_audio(chat_id=last_update(get_updates_json(url))['message']['from']['id'], audio=open('./audio.mp3', 'rb'))
    while(True):
        if(mssg_ID != last_update(get_updates_json(url))['message']['message_id']):
            break