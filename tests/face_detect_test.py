import cv2
import torch

from detect_face import load_model, detect_one


class TestFaceDetect:
    def test_face_detect(self):
        weights = 'yolov5_face/weights/yolov5n-0.5.pt'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(weights, device)

        img = cv2.imread('tests/assets/screen.jpg')

        image, coords = detect_one(model, img, device) # coords in format [x0, y0, x1, y1]
        expected = torch.Tensor([[899., 170., 943., 222.],[87., 425., 161., 494.]]).to(device)
        assert torch.equal(coords, expected)
        
