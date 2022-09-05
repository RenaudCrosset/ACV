import cv2
from filters import filters
from filters import filters_mediapipe
import numpy as np


filters_dic = {
ord('1'):filters.mirror,
ord('2'):filters.glow,
ord('3'):filters.sepia,
ord('4'):filters.b_and_w,
ord('5'):filters.xRay,
ord('6'):filters.cartoon,
ord('7'):filters.drawing,
ord('8'):filters.thermal_cam,
ord('9'):filters.quad_cam,
ord('p'):filters.TV,
ord('o'):filters.wave,
ord('n'):filters_mediapipe.red_nose,
ord('b'):filters_mediapipe.carnival
}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

filter_choice = []

while cap.isOpened():

    ret, frame = cap.read()
    if ret:

        key = cv2.waitKey(10) & 0xFF
        if key in filters_dic.keys():
            filter_choice.append(filters_dic[key])

        if key==ord('0'):
            filter_choice = []
        else:
            for filter in filter_choice:
                frame = filter(frame)
    
        cv2.imshow('image', cv2.flip(frame, 1))

        if key == ord('q'):
            break
    else:
        break
 
cap.release()
cv2.destroyAllWindows()


# filter_choice = None

# while cap.isOpened():

#     ret, frame = cap.read()
#     if ret:

#         key = cv2.waitKey(10) & 0xFF
#         if key in filters_dic.keys():
#             filter_choice = filters_dic[key]

#         if ((filter_choice==None) or (key==ord('0'))):
#             cv2.imshow('image', cv2.flip(frame, 1))
#             filter_choice = None
#         else:
#             cv2.imshow('image', cv2.flip(filter_choice(frame), 1))

#         if key == ord('q'):
#             break
#     else:
#         break
 
# cap.release()
# cv2.destroyAllWindows()