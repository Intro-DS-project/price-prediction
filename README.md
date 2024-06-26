# price-prediction

## How to run

- Install libraries: ```pip install -r requirements.txt```

### Train model:

```python train.py```

This will:
- Load currenr model and check with today's data
- If the error > threshold, re-train the model
- Re-train stagegy: train data includes old data (within 30 days) and a part of new data (today). Other new data is validation data. Finetune model hyperparameters using Ray Tune

### Run Prediction API
```uvicorn main:app --host <YOUR_HOST_IP> --port <YOUR_HOST_PORT>```

- API Endpoint: /predict (POST)
- Request Body example:
```
{
  "area": 20,
  "num_bedroom": 0,
  "num_diningroom": 0,
  "num_kitchen": 0,
  "num_toilet": 0,
  "street": "Đại La",
  "ward": "Trương Định",
  "district": "Hai Bà Trưng"
}
```

- Response example:
```
{
  "success": true,
  "price": 3.038297414779663
}
```

**Update**: API Endpoint: /loss (GET)
- Response example:
```
{
  "success": true,
  "loss": 0.9340711236000061
}
```


### Run prediction API with Docker
- Build Docker image:

```docker build -t <image-name> .```

- Run Docker container (expose port **8000**)

```docker run -p 8000:8000 <image-name>```