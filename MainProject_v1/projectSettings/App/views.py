from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse, HttpResponse
##########################
import json
import random
import torch
from MainProject_v1.projectSettings.App.ChatBot.model import NeuralNet
from MainProject_v1.projectSettings.App.ChatBot.nltk_utils import word_bag, tokenize


import os
dir_path = os.path.dirname(os.path.realpath(__file__))

##########################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

intensts_path = os.path.join(dir_path,'ChatBot','intents.json')
with open(intensts_path, "r") as f:
    intents = json.load(f)


FILE = os.path.join(dir_path,'ChatBot','data.pth')
data = torch.load(FILE)


# Create your views here.
@api_view(["POST"])
def test(request):
    sentence = str(request.data)

    try:
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]

        all_words = data["all-wards"]
        tags = data["tags"]
        model_state = data["model_state"]

        model = NeuralNet(input_size, hidden_size, output_size)
        model.load_state_dict(model_state)
        model.eval()

        sentence = tokenize(sentence)
        x = word_bag(sentence, all_words)
        x = x.reshape(1, x.shape[0])
        x = torch.from_numpy(x)

        output = model(x)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    result = str(random.choice(intent["responses"]))
                    #JsonResponse(result, safe=False)
                    return HttpResponse(result)
        else:
            result = "Sorry I'm unable to understand"
            #JsonResponse(result, safe=False)
            return HttpResponse(result)
    except ValueError as e:
        Response(e.args[0], status.HTTP_400_BAD_REQUEST)
