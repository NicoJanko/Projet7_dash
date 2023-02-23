import requests

def request_test(status=False):
    api_uri = 'http://pad-app.herokuapp.com'
    response = requests.get(api_uri +'/test', json={'test':100001})
    status_code = response.status_code
    response_test = response.json()
    if status:
        return status_code
    else:
        return response_test

def request_pred():
    api_uri = 'http://pad-app.herokuapp.com'
    response_pred = requests.get(api_uri + '/predict', json={'client_id' : 100001}).json()
    return response_pred

def test_status_code():
    status_code = request_test(status=True)
    assert status_code == 200

def test_db_1():
    response_test = request_test()
    assert response_test['client_db'] == [[568800.0]]

def test_db_2():
    response_test = request_test()
    assert response_test['raw_data'] == [['F']]


def test_pred():
    response_pred = request_pred()
    assert response_pred['prediction'] == 0

def test_proba():
    response_pred = request_pred()
    assert response_pred['probability'] == 15



