from asyncio import sleep
import random
import requests


url = "http://localhost:9696/predict"
client = {"job": "retired", "duration": 445, "poutcome": "success"}

# get random values for retired, working, unemployed
status = ["retired", "working", "unemployed"]


while True:
    sleep(.1)
    duration = random.randint(400, 500)
    client["duration"] = duration
    client["job"] = random.choice(status)
    response = requests.post(url, json=client).json()
    print(response)
