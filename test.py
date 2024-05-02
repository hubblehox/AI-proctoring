import cv2

cap = cv2.VideoCapture('data/op.mp4')

while True:
    status, img = cap.read()
    if status == True:
        cv2.imshow('img', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()