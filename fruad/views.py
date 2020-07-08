from django.shortcuts import render
import json
import joblib
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
# Create your views here.

@csrf_exempt
def predict(request):
    if request.POST is not None:

        model = joblib.load('model.sav')
        scaler = joblib.load('scaler.sav')

        time = float(request.POST['time'])
        amount = float(request.POST['amount'])
        time = scaler.fit_transform(np.array([time]))
        amount = scaler.fit_transform(np.array([amount]))
        v1 = float(request.POST['v1'])
        v2 = float(request.POST['v2'])
        v3 = float(request.POST['v3'])
        v4 = float(request.POST['v4'])
        v5 = float(request.POST['v5'])
        v6 = float(request.POST['v6'])
        v7 = float(request.POST['v7'])
        v8 = float(request.POST['v8'])
        v9 = float(request.POST['v9'])
        v10 = float(request.POST['v10'])
        v11 = float(request.POST['v11'])
        v12 = float(request.POST['v12'])
        v13 = float(request.POST['v13'])
        v14 = float(request.POST['v14'])
        v15 = float(request.POST['v15'])
        v16 = float(request.POST['v15'])
        v17 = float(request.POST['v17'])
        v18 = float(request.POST['v18'])
        v19 = float(request.POST['v19'])
        v20 = float(request.POST['v20'])
        v21 = float(request.POST['v21'])
        v22 = float(request.POST['v22'])
        v23 = float(request.POST['v23'])
        v24 = float(request.POST['v24'])
        v25 = float(request.POST['v25'])
        v26 = float(request.POST['v26'])
        v27 = float(request.POST['v27'])
        v28 = float(request.POST['v28'])

        pred = model.predict(np.array([time, amount, v1, v2, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
                                       v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28]))



        return JsonResponse(pred)