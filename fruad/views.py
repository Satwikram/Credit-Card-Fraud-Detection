from django.shortcuts import render
import json
import joblib
import numpy as np

# Create your views here.

def predict(request):
    if request.json is not None:

        model = joblib.load('model.sav')
        scaler = joblib.load('scaler.sav')

        data = json.loads(request.body)
        print(data)

