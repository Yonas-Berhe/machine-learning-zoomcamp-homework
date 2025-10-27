import requests

url = 'http://localhost:9090/predict2'
# url = 'https://mlzoomcamp-flask-uv.fly.dev/predict'

customer = {
   "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0    
}

response = requests.post(url, json=customer)

predictions = response.json()


if predictions['convert']:
    print('customer is likely to convert, send promo')
else:
    print('customer is not likely to convert, do not send promo')