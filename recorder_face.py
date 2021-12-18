import itertools
import os
import io
import sys
import time
import json
import random
import base64
import struct

import cv2
import torch
import numpy as np
import redis
import tornado
from PIL import Image, ImageDraw
sys.path.insert(0, './yolov5_face')

from detect_face import load_model, detect_one


# Retrieve command line arguments.
WIDTH = None if len(sys.argv) <= 1 else int(sys.argv[1])
HEIGHT = None if len(sys.argv) <= 2 else int(sys.argv[2])

# Create video capture object, retrying until successful.
MAX_SLEEP = 5.0
CUR_SLEEP = 0.1


def create_videostream():
    while True:
        cap = cv2.VideoCapture('bus.avi')
        if cap.isOpened():
            break
        print(f'not opened, sleeping {CUR_SLEEP}s')
        time.sleep(CUR_SLEEP)
        if CUR_SLEEP < MAX_SLEEP:
            CUR_SLEEP *= 2
            CUR_SLEEP = min(CUR_SLEEP, MAX_SLEEP)
            continue
        CUR_SLEEP = 0.1

    # Create client to the Redis store.
    store = redis.Redis()

    # Set video dimensions, if given.
    if WIDTH:
        cap.set(3, WIDTH)
    if HEIGHT:
        cap.set(4, HEIGHT)

    # Init model for faces
    weights = 'yolov5_face/weights/yolov5n-0.5.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_model = load_model(weights, device)

    for count in itertools.count(1):
        _, orgimg = cap.read()
        thres = 0
        if orgimg is None:
            time.sleep(0.5)
            thres += 1
            if thres > 10:
                break
            continue

        clients = detect_faces(face_model, orgimg, device)
        
        _, image = cv2.imencode('.jpg', orgimg)

        store.set('coords', json.dumps(clients))
        store.set('image', np.array(image).tobytes())
        store.set('image_id', os.urandom(4))
        print(count)
        # print(clients)

def detect_faces(model, orgimg, device):
    s = time.time()
    image, coords = detect_one(model, orgimg, device)
    print(coords.tolist())
    f = time.time()
    print(f-s)

    clients = list()

    for idx, coord in enumerate(coords):
        clients.append({'id': idx, 'coords': coord.tolist(), 'datetime': random_date("1/1/2018 1:30 PM", "1/1/2022 4:50 AM", random.random()), 'rating': random.uniform(1, 5)})

    return clients


def random_date(start, end, prop):
    return str_time_prop(start, end, '%m/%d/%Y %I:%M %p', prop)


def str_time_prop(start, end, time_format, prop):
    """Get a time at a proportion of a range of two formatted times.
    start and end should be strings specifying times formatted in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, time_format))
    etime = time.mktime(time.strptime(end, time_format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(time_format, time.localtime(ptime))
    

if __name__ == '__main__':
    create_videostream()