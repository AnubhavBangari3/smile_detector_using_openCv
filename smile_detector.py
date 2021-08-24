
import cv2

face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector=cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detector=cv2.CascadeClassifier('haarcascade_eye.xml')

webcam=cv2.VideoCapture(0)

while True:
    successful_frame_read,frame=webcam.read()
    if not successful_frame_read:
        break
    grayscaled_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face=face_detector.detectMultiScale(grayscaled_frame)  

    for x,y,w,h in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(10,200,0),2)
        #trying to get face
        the_face=frame[y:y+h , x:x+w]

        face_grayscale=cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)
        #detect smile in face
        smile=smile_detector.detectMultiScale(face_grayscale,1.7,20)

        eye=eye_detector.detectMultiScale(face_grayscale,1.5,10)

        for (x_, y_, w_, h_) in smile:
            cv2.rectangle(the_face,(x_ , y_),(x_ + w_ , y_ + h_),(10,0,250),2)
        for (x1,y1,w1,h1) in eye:
            cv2.rectangle(the_face,(x1,y1),(x1+w1,y1+h1),(255,0,0),1)
    smile=smile_detector.detectMultiScale(face_grayscale,1.7,20)
    #label this face as smiling
    if len(smile)>0:
        cv2.putText(frame,'smiling',(x,y+h+40),fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,255,255))

    cv2.imshow("Smile",frame)
    key=cv2.waitKey(1)
    if key==81 or key==113:
        break
webcam.release()
cv2.destroyAllWindows()

print("Working")