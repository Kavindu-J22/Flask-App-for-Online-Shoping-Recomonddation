import requests
import json

# URL of the Flask app's prediction endpoint
url = 'http://127.0.0.1:5000/predict'

# Sample input data
data = {
    "title": "Amazing product!",
    "review_text": "I loved the quality of this product. Highly recommend!"
}

# Convert data to JSON format
headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json.dumps(data), headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Get the JSON response from the server
    result = response.json()
    print("Prediction:", result['prediction'])
else:
    print(f"Failed to get response. Status code: {response.status_code}")
