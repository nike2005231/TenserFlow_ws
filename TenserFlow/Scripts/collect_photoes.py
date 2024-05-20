import numpy as np
import cv2
import os
import datetime

if __name__ == "__main__":
    try:
        os.mkdir('images')
    except:
        pass
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            break
        elif pressed_key == ord('s'):
            now = datetime.datetime.now()
            current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
            filename = '%s.png' % current_time
            filename_with_path = 'images/' + filename
            print(cv2.imwrite(filename_with_path, frame))
    cap.release()
    cv2.destroyAllWindows()
