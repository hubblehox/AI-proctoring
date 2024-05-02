import cv2
import pandas as pd
from face_verification import Face_verification
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
from eye_tracker_m import *
import time
import pyaudio
import math
import wave
from utilis import percentage_of_time, softmax
from object_detection import Object_Detection
from headpose_detection import get_2d_points, draw_annotation_box, head_pose_points
import random
#from emotion_detection import face_detector_ep
logging_lst = []


def main_func(file_path,initial = 'data/souvik.jpg'):
    face_verfication_counter = 0
    face_model = get_face_detector()
    landmark_model = get_landmark_model()
    left = [36, 37, 38, 39, 40, 41]
    right = [42, 43, 44, 45, 46, 47]
    emotion_dict = {'Surprise': 0, 'Neutral': 0, 'Anger': 0, 'Happy': 0, 'Sad': 0}
    cap = cv2.VideoCapture(file_path)
    ret, img = cap.read()
    size = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX 

    thresh = img.copy()
    model_points = np.array([(0.0, 0.0, 0.0),
                             (0.0, -330.0, -65.0),
                             (-225.0, 170.0, -135.0),
                             (225.0, 170.0, -135.0),
                             (-150.0, -150.0, -125.0),
                             (150.0, -150.0, -125.0)])
    
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype = "double")
    
    
    # Define the codec and create VideoWriter object
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('data/op.mp4', fourcc, fps, (width, height))



    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.moveWindow('image', 1920, 1080)
    kernel = np.ones((9, 9), np.uint8)
    audio = pyaudio.PyAudio()
    cv2.createTrackbar('threshold', 'image', 75, 255, none)
    no_person_counter = multiple_person_counter = 0
    left_look_counter = right_look_counter = 0
    counter = 0
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (50, 50)  
    fontScale = 1
    color = (255, 0, 0) 
    thickness = 2
    frame_id = 0
    
    start = time.time()
    while True:
        lst = []
        status = True
        none_availability, multiple_availability, status_ok = np.nan, np.nan, np.nan
        prohibited_items_counter, looking_away, looking_left, looking_right = np.nan, np.nan, np.nan, np.nan
        verified = np.nan
        location = np.nan
        

        frame_id += 1
        location = f"images/frame{str(frame_id)}.jpg"
        
        ret, img = cap.read()
        
        if ret:
            #cv2.imshow('Webcam', img)
            rects = find_faces(img, face_model)
            
            if len(rects) == 0:
                print("None available")
                text = "None available"
                cv2.putText(img, text, org, font,  fontScale, color, thickness, cv2.LINE_AA) 
                status = False
                none_availability = 1
                no_person_counter += 1
            elif len(rects) > 1:
                multiple_person_counter += 1
                print("Multiple available")
                text = "Multiple face detected"
                cv2.putText(img, text, org, font,  fontScale, color, thickness, cv2.LINE_AA) 
                status = False
                multiple_availability = 1
        else:
            break

        try:
            for rect in rects:
                shape = detect_marks(img, landmark_model, rect)
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask, end_points_left = eye_on_mask(mask, left, shape)
                mask, end_points_right = eye_on_mask(mask, right, shape)
                mask = cv2.dilate(mask, kernel, 5)

                eyes = cv2.bitwise_and(img, img, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]
                mid = int((shape[42][0] + shape[39][0]) // 2)
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                threshold = cv2.getTrackbarPos('threshold', 'image')
                _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
                thresh = process_thresh(thresh)

                eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
                eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)
                # print(eyeball_pos_left,eyeball_pos_right)

                if len(rects) == 1:
                    boolean, prohibited_items = Object_Detection(img)
                    if boolean:
                        text = f"Prohibited items detected : {prohibited_items}"
                        cv2.putText(img, text, org, font,  fontScale, color, thickness, cv2.LINE_AA) 
                        status = False
                        prohibited_items_counter = 1
                    if (eyeball_pos_left is None) and (eyeball_pos_right is None):
                        print("Looking Away")
                        text = "Looking Away"
                        cv2.putText(img, text, org, font,  fontScale, color, thickness, cv2.LINE_AA) 
                        status = False
                        looking_away = 1
                    elif eyeball_pos_left is None:
                        right_look_counter += 1
                        print("Looking Right")
                        text = "Looking Right"
                        cv2.putText(img, text, org, font,  fontScale, color, thickness, cv2.LINE_AA) 
                        status = False
                        looking_left = 1
                    elif eyeball_pos_right is None:
                        left_look_counter += 1
                        print("Looking Left")
                        text = "Looking Left"
                        cv2.putText(img, text, org, font,  fontScale, color, thickness, cv2.LINE_AA)
                        status = False
                        looking_right = 1
                    #state = face_detector_ep(image=img)
                    #emotion_dict[state] += 1
                
                    #print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)
                    #status = False
        except:
            pass
        if status:
            status_ok = 1
            cv2.putText(img, "Status : OK", org, font,  fontScale, (0,255,0), thickness, cv2.LINE_AA)
        cv2.imwrite(location, img)
        choice = random.choice(range(0,100))
        if choice < 3:
            verified = Face_verification(initial, location)
        cv2.imshow('Window', img)
        cv2.imshow("image", thresh)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(img)
        counter += 1
        
        lst = [frame_id, location, status_ok, none_availability, multiple_availability,
               prohibited_items_counter, looking_away, looking_right, looking_left, verified]
        logging_lst.append(lst)
    stop = time.time()
    session = stop-start
    no_person_duration, no_person_percentage = percentage_of_time(session_duration=session,
                                                                  total_frames=counter,
                                                                  frames=no_person_counter)

    multiple_person_duration, multiple_person_percentage = percentage_of_time(session_duration=session,
                                                                  total_frames=counter,
                                                                  frames=multiple_person_counter)

    right_look_duration, right_look_percentage = percentage_of_time(session_duration=session,
                                                                 total_frames=counter,
                                                                 frames=right_look_counter)

    left_look_duration, left_look_percentage = percentage_of_time(session_duration=session,
                                                                 total_frames=counter,
                                                                 frames=left_look_counter)

    print(f"Session duration : {round(session/60,2)} minutes")
    print(f"None available : {no_person_percentage}%, "
          f"Multiple available : {multiple_person_percentage}%, "
          f"Looking right : {right_look_percentage}%, "
          f"Looking left : {left_look_percentage}%")

    #emotion_dict = softmax(emotion_dict)
    #print(emotion_dict)
    df = pd.DataFrame(logging_lst, 
                      columns=['frame_id','location','status','none','multiple','prohibited',
                               'looking_away','looking_right','looking_left','verified'])
    df.to_csv("data/logging.csv", index=False)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return {'session_duration': f"{round(session/60,2)} minutes",
            "availability": {"None available": f"{no_person_percentage}%",
                              "Multiple available": f"{multiple_person_percentage}%",
                              "Looking right": f"{right_look_percentage}%",
                              "Looking left": f"{left_look_percentage}%"}}


if __name__ == "__main__":
    #file_path = input("Enter the file path:\n")
    results = main_func(file_path='data/video.mp4', initial = 'data/souvik.jpg')
