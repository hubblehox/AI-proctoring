import torch
import os
os.environ['YOLO_VERBOSE'] = "False"


def Object_Detection(frame, prohibited_items : list = ['cell phone']):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    results = model([frame])
    items = list(results.pandas().xyxy[0].name.unique())
    prohibited_items_found = []
    boolean = False
    for i in items:
        if i in prohibited_items:
            prohibited_items_found.append(i)
            boolean = True
            
    return boolean, prohibited_items_found

if __name__ == "__main__":
    path = input('Enter the path: \n')
    alert, items = Object_Detection(frame=path)
    print(alert, items)