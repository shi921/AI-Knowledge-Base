"""pytest projects/04_flask_iris_api/test_api.py"""
from app import create_app

def test_health():
    client=create_app().test_client()
    assert client.get('/health').json['status']=='ok'

def test_predict():
    client=create_app().test_client()
    reply=client.post('/predict',json={'sl':5.1,'sw':3.5,'pl':1.4,'pw':0.2})
    assert reply.status_code==200
    assert reply.json['flower'].startswith('Iris-')

def test_validation():
    client=create_app().test_client()
    assert client.post('/predict',json={'sl':'NaN'}).status_code==400
