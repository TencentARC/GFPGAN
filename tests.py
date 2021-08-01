from fastapi.testclient import TestClient
from pathlib import Path
import requests, time

from requests.models import HTTPError

from main import app
import cv2 as cv
import numpy as np
from main import is_url_image

client = TestClient(app)
url = "http://127.0.0.1:8000"


test_image_path = Path("./inputs/whole_imgs/00.jpg")
def test_create_upload_file():
    print("Staring test_create_upload_file test...")

    response = client.post('/image_upload/', 
                            files = {"file": (test_image_path.name, 
                            open(test_image_path, 'rb'), 
                            "image/jpeg")}
                            )

    assert response.status_code == 200
    print(response.content)
#test_create_upload_file()

def sleep(timeout, retry=3):
    def the_real_decorator(function):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < retry:
                try:
                    value = function(*args, **kwargs)
                    if value is None:
                        return True
                except:
                    print(f'Sleeping for {timeout} seconds')
                    time.sleep(timeout)
                    retries += 1
            return False
        return wrapper
    return the_real_decorator
    

@sleep(3, retry =3)
def does_image_exists(url):
    try:
        ret_val = is_url_image(url)
        if ret_val is False:
            print('Image not found. Trying again...')
            raise HTTPError
        
    except HTTPError as e:
        print('HTTP error:{e}')
        raise HTTPError
    else:
        print("Image found in website")
        
    
def test_inference_with_cloudinary(url):

    #Part 1 of test
    response = client.post('/inference_from_cloudinary', headers={'img_link': url})
    assert response.status_code == 200
    print("subtest #1 starting.")

    #Part 2 of test 
    #Build URL for checking
    print("subtest #2 Starting.") 
    img_id = response.json().get('img_id')
    check_img_url = img_id+'_enhanced.jpg'
    url_check = url[:url.rfind('/')]
    url_check = url_check + '/' + check_img_url
    print("subtest #2.1 Starting.")
    ret_val = does_image_exists(url_check)
    assert  ret_val == True
    print("Test test_inference_with_cloudinary: Done")

#this test is meant to fail to False
def test_does_image_exist():
    print("Starting does_image_exists Test")
    non_existent_image_url = "https://res.cloudinary.com/dydx43zon/image/upload/v1627664136/fucenobitest.jpg"
    exists = does_image_exists(non_existent_image_url)
    assert exists == False
    print("Test test_does_image_exist: Done")

    
print("Strarting tests...")
test_does_image_exist()

test_url = "https://res.cloudinary.com/dydx43zon/image/upload/v1627664136/ffuckingkenobitest.jpg"
test_inference_with_cloudinary(test_url)

print("Tests have  Passed")