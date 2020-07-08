from django.shortcuts import render
import json
import joblib
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
# Create your views here.

@csrf_exempt
def predict(request):
    if request.data is not None:

        model = joblib.load('model.sav')
        scaler = joblib.load('scaler.sav')

        time = float(request.data['time'])
        amount = float(request.data['amount'])
        time = scaler.fit_transform(np.array([time]))
        amount = scaler.fit_transform(np.array([amount]))
        v1 = float(request.data['v1'])
        v2 = float(request.data['v2'])
        v3 = float(request.data['v3'])
        v4 = float(request.data['v4'])
        v5 = float(request.data['v5'])
        v6 = float(request.data['v6'])
        v7 = float(request.data['v7'])
        v8 = float(request.data['v8'])
        v9 = float(request.data['v9'])
        v10 = float(request.data['v10'])
        v11 = float(request.data['v11'])
        v12 = float(request.data['v12'])
        v13 = float(request.data['v13'])
        v14 = float(request.data['v14'])
        v15 = float(request.data['v15'])
        v16 = float(request.data['v15'])
        v17 = float(request.data['v17'])
        v18 = float(request.data['v18'])
        v19 = float(request.data['v19'])
        v20 = float(request.data['v20'])
        v21 = float(request.data['v21'])
        v22 = float(request.data['v22'])
        v23 = float(request.data['v23'])
        v24 = float(request.data['v24'])
        v25 = float(request.data['v25'])
        v26 = float(request.data['v26'])
        v27 = float(request.data['v27'])
        v28 = float(request.data['v28'])

        pred = model.predict(np.array([time, amount, v1, v2, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
                                       v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28]))



        return JsonResponse(pred)