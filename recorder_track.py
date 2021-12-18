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
sys.path.insert(0, './yolov5')

from yolov5.utils.downloads import attempt_download
from yolov5.utils.torch_utils import select_device
from yolov5.models.common import DetectMultiBackend

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from track import detect, draw_boxes


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

    store = redis.Redis()

    if WIDTH:
        cap.set(3, WIDTH)
    if HEIGHT:
        cap.set(4, HEIGHT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_weights = 'yolov5s.pt'
    imgsz = [640]

    # initialize deepsort
    deep_sort_weights = 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
    cfg = get_config()
    cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Init model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device)
    names = model.module.names if hasattr(model, 'module') else model.names

    for count in itertools.count(1):
        _, orgimg = cap.read()
        thres = 0
        if orgimg is None:
            time.sleep(0.5)
            thres += 1
            if thres > 10:
                break
            continue

        outputs, confs = detect(orgimg, imgsz, count, model, deepsort, device)

        clients = [out.tolist() for out in outputs]
        image = draw_boxes(outputs, confs, orgimg, names)
        _, image = cv2.imencode('.jpg', image)

        store.set('coords', json.dumps(clients))
        store.set('image', np.array(image).tobytes())
        store.set('image_id', os.urandom(4))
        print(count)
        # print(clients)


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