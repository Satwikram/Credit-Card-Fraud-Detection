from django.shortcuts import render
import json
import joblib
import numpy as np
from django.http import JsonResponse
# Create your views here.

def predict(request):
    if request.json is not None:

        model = joblib.load('model.sav')
        scaler = joblib.load('scaler.sav')

        time = float(request.json['time'])
        amount = float(request.json['amount'])
        time = scaler.fit_transform(np.array([time]))
        amount = scaler.fit_transform(np.array([amount]))
        v1 = float(request.json['v1'])
        v2 = float(request.json['v2'])
        v3 = float(request.json['v3'])
        v4 = float(request.json['v4'])
        v5 = float(request.json['v5'])
        v6 = float(request.json['v6'])
        v7 = float(request.json['v7'])
        v8 = float(request.json['v8'])
        v9 = float(request.json['v9'])
        v10 = float(request.json['v10'])
        v11 = float(request.json['v11'])
        v12 = float(request.json['v12'])
        v13 = float(request.json['v13'])
        v14 = float(request.json['v14'])
        v15 = float(request.json['v15'])
        v16 = float(request.json['v15'])
        v17 = float(request.json['v17'])
        v18 = float(request.json['v18'])
        v19 = float(request.json['v19'])
        v20 = float(request.json['v20'])
        v21 = float(request.json['v21'])
        v22 = float(request.json['v22'])
        v23 = float(request.json['v23'])
        v24 = float(request.json['v24'])
        v25 = float(request.json['v25'])
        v26 = float(request.json['v26'])
        v27 = float(request.json['v27'])
        v28 = float(request.json['v28'])

        pred = model.predict(np.array([time, amount, v1, v2, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
                                       v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28]))



        return JsonResponse(pred)