from cProfile import label
from tkinter import Frame
import torch
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import numpy as np

class CharacterDetector:
    def __init__(self, weights_path='weights/best_character.pt'):
        self.conf_thres=0.7
        self.iou_thres=0.45 
        self.max_det=1000  
        self.classes=None  
        self.agnostic_nms=False  
        self.device = select_device("")
        self.model = DetectMultiBackend(weights_path, 
                                        device=self.device, 
                                        dnn=False, 
                                        data='data/character.yaml', 
                                        fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size((640, 640), s=self.stride)  
        self.model.warmup(imgsz=(1 , 3, *self.imgsz))  
    
    def getCharacter(self, image):
            
        image_original = image
        
        image_size = image.shape
        image = letterbox(image, self.imgsz, stride=self.stride, auto=True)[0]
        image = image.transpose((2, 0, 1))[::-1]  
        image = np.ascontiguousarray(image)
        image_torch = torch.from_numpy(image).to(self.device)
        image_torch = image_torch.half() if self.model.fp16 else image_torch.float() 
        image_torch /= 255 
        if len(image_torch.shape) == 3: image_torch = image_torch[None]  
        pred = self.model(image_torch, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, 
                                self.classes, self.agnostic_nms, max_det=self.max_det)
        
        output = []
        for i, det in enumerate(pred):
            det[:, :4] = scale_coords(image_torch.shape[2:], det[:, :4], image_size).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                w = y2-y1
                h = x2-x1
                bbox = [x1, y1, x2 - x1, y2 - y1]
                # cls = int(cls)
                if cls == 0: # เอาแต่ตัวอักษรหรือเลข ยังไม่เอาจังหวัด
                    output.append([bbox , conf , image_original[y1:y1+w , x1:x1+h]])
                
        return output
    