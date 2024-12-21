from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch
import numpy as np
import cv2

model_path = 'google/owlvit-base-patch16'

label_texts = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']

class DeepLearningProcessor:
    def __init__(self):
        self._proc = OwlViTProcessor.from_pretrained(model_path)        
        self._model = OwlViTForObjectDetection.from_pretrained(
            model_path,
            ignore_mismatched_sizes=True,
        )

    def process(self, img_in: cv2.typing.MatLike):
        img_out = img_in.copy()

        inp = self._proc(text=label_texts, images=img_in, return_tensors='pt')
        out = self._model(**inp)

        h, w, c = img_in.shape
        target_sizes = [(h, w)]
        res = self._proc.post_process_object_detection(outputs=out, threshold=0.1, target_sizes=target_sizes)

        labels, boxes = res[0]['labels'], res[0]['boxes']
        for label_idx, box in zip(labels, boxes):
            label_text = label_texts[label_idx]
            box: torch.Tensor = box.round().int()
            [x1, y1, x2, y2] = box.tolist()
            print(f'Detected {label_text} at ({x1}, {y1}) : ({x2}, {y2})')

            cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)

            text_x = x1
            text_y = y1 - 10

            if text_y < 10:
                text_y = y2 - 10
                text_x = text_x + 20

            cv2.putText(img_out, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return img_out
