import cv2 as cv

cam=cv.imread("elon.jpg")

eye= cv.CascadeClassifier("haarcascade_eye.xml")
face = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
grey= cv.cvtColor(cam, cv.COLOR_BGR2GRAY)

eye_detect = eye.detectMultiScale(grey,1.3,5)
face_detect = face.detectMultiScale(grey,1.3,5)
for (x,y,w,h) in eye_detect:
    cv.rectangle(cam, (x,y),(x+w,y+h), (0,233,0),2)
    cv_warna = cam[y:y+h,x:+w]
    cv_grey = grey[y:y+h,x:+w]

for (x1,y1,w1,h1) in face_detect:
    cv.rectangle(cam, (x1,y1),(x1+w1,y1+h1), (0,233,0),2)
    cv_warna = cam[y1:y1+h1,x1:+w1]
    cv_grey = grey[y1:y1+h1,x1:+w1]

cv.imshow("image", cam)
cv.waitKey(0)
cv.destroyAllWindows()