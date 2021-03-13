import cv2
import os
from os import path
from picamera import PiCamera
from picamera.array import PiRGBArray

cam = PiCamera()
cam.resolution = (512, 304)
cam.framerate = 10
rawCapture = PiRGBArray(cam, size=(512, 304))

name = input("Give person name: ")
if not path.exists("dataset/"+name):
    os.mkdir("dataset/" + name)
img_counter = len(os.listdir("dataset/" + name))

while True:
    for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        
        image = frame.array
        cv2.imshow("Press Space to take a photo", image)
        rawCapture.truncate(0)
    
        k = cv2.waitKey(1)
        rawCapture.truncate(0)
        
        if k%256 == 27: # ESC pressed
            break

        if k%256 == 13: # ESC pressed
            print("hi")
        elif k%256 == 32:
            # SPACE pressed
            img_name = "dataset/"+ name +"/image_{}.jpg".format(img_counter)
            img_counter += 1
            cv2.imwrite(img_name, image)
            print(f"{img_name} written!")
            
            
    if k%256 == 27:
        print("Escape hit, closing...")
        break

cv2.destroyAllWindows()
