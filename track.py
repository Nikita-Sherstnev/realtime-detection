# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')
print(sys.path)

from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import LOGGER, check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.augmentations import letterbox
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np


def detect(img0, imgsz, frame_idx, model, deepsort, device):
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    classes = [0] # class 0 is person, 1 is bycicle, 2 is car... 79 is oven

    # Initialize
    half = True
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None

    # Dataloader
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    # Padded resize
    img = letterbox(img0, imgsz, stride=stride, auto=True)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    
    t1 = time_sync()
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    augment = True
    pred = model(img, augment=augment, visualize=False)
    t3 = time_sync()
    dt[1] += t3 - t2

    # Apply NMS
    conf_thres = 0.4
    iou_thres = 0.4
    agnostic_nms = True
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=1000)
    dt[2] += time_sync() - t3

    s = 'image ' # for logger

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        seen += 1

        im0, frame = img0.copy(), frame_idx

        s += '%gx%g ' % img.shape[2:]  # print string

        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]

            # pass detections to deepsort
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
        else:
            deepsort.increment_ages()

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    print(outputs)
    return outputs, confs


def draw_boxes(outputs, confs, im0, names):
    # draw boxes for visualization
    if len(outputs) > 0:
        for j, (output, conf) in enumerate(zip(outputs, confs)): 
            
            bboxes = output[0:4]
            id = output[4]
            cls = output[5]

            c = int(cls)  # integer class
            label = f'{id} {names[c]} {conf:.2f}'

            # to MOT format
            bbox_left = output[0]
            bbox_top = output[1]
            bbox_w = output[2] - output[0]
            bbox_h = output[3] - output[1]
            # Write MOT compliant results to file
            # with open(txt_path, 'a') as f:
            #     f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,
            #                                     bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

        annotator = Annotator(im0, line_width=2, pil=not ascii)
        annotator.box_label(bboxes, label, color=colors(c, True))
        im0 = annotator.result()
        return im0


def detection():
    yolo_weights = 'yolov5s.pt'
    imgsz = [1280]

    MAX_SLEEP = 5.0
    CUR_SLEEP = 0.1
    
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
    device = select_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = DetectMultiBackend(yolo_weights, device=device)

    for i in range(5):
        _, orgimg = cap.read()
        with torch.no_grad():
            outputs, confs = detect(orgimg, imgsz, i, model, deepsort, device)
            print(confs)

if __name__ == '__main__':
    detection()