import requests

url = "http://127.0.0.1:8000/infer"
input_vector = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

response = requests.post(url, json={"input_vector": input_vector})
print(response.json())
