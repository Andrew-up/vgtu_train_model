import json

import requests
from model.model_history import ModelHistory
API = 'http://localhost:5000/api/model_cnn/'

def get_model_history():
    r = requests.get(API+'train/')
    print(r.json())
    return 0

def update_model_history(data: ModelHistory):
    url_string = f'{API}update/'
    data_json = json.dumps(data.__dict__)
    r = requests.post(url_string, data=data_json, headers={"Content-Type": "application/json"})
    # print(r.request.headers)
    # print(r.request.body)
    # print(r.request.hooks)
    # print(r.request.method)
    # # print(r.url)
    print(r.text)



if __name__ == '__main__':
    # get_model_history()
    m = ModelHistory()
    m.id = 1
    m.status = 'test1ddd23'
    update_model_history(m)
