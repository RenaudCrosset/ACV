import mediapipe as mp
import numpy as np
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_face_mesh = mp.solutions.face_mesh
effect = 'nez_rouge' # 'nez_rouge' ou 'crotte_nez' ou 'crotte_nez_v2'

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
max_num_faces = 5

with mp_face_mesh.FaceMesh(max_num_faces=max_num_faces,
                           refine_landmarks=True, 
                           min_detection_confidence=0.5, 
                           min_tracking_confidence=0.5
                          )as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue  # If loading a video, use 'break' instead of 'continue'
        print("ok")
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            color = (0, 0, 255)
            height, width, _ = image.shape
            if effect == 'nez_rouge':
                for nb_face in range(len(results.multi_face_landmarks)):
                    nose = results.multi_face_landmarks[nb_face].landmark[4]
                    cv2.circle(image, (int(nose.x*width), int(nose.y*height)), int(-400*nose.z+1), color, -1)
            elif effect == 'crotte_nez':
                obj = cv2.imread("data/goutte.jpg") #morve_square.jpg")
                for nb_face in range(len(results.multi_face_landmarks)):
                    under_nose = results.multi_face_landmarks[nb_face].landmark[79]
                    scalefactor = 1 - 2*under_nose.z
                    obj = cv2.resize(obj, None, fx=1, fy=scalefactor, interpolation=cv2.INTER_LANCZOS4)
                    mask = 255 * np.ones(obj.shape, obj.dtype)
                    width, height, channels = image.shape
                    center = (int(under_nose.x*height), int(under_nose.y*width) + obj.shape[0]//2)
                try:
                    image = cv2.seamlessClone(obj, image, mask, center, cv2.MIXED_CLONE)
                except:
                    pass
          
            elif effect == 'crotte_nez_v2':
                obj = cv2.imread("data/goutte.jpg") #morve_square.jpg")
                obj_h, obj_w, _ = obj.shape
                for nb_face in range(len(results.multi_face_landmarks)):
                    try:
                        under_nose = results.multi_face_landmarks[nb_face].landmark[79]
                        image_crop = image[int(under_nose.y*height):int(under_nose.y*height)+obj_h,
                                           int(under_nose.x*width)-obj_w//2:int(under_nose.x*width)+obj_w//2, :]
                        image_crop_fusion = cv2.addWeighted(image_crop, 0.5, obj, 0.5, 0)
                        image[int(under_nose.y*height):int(under_nose.y*height)+obj_h,
                              int(under_nose.x*width)-obj_w//2:int(under_nose.x*width)+obj_w//2, :] = image_crop_fusion
                    except:
                        pass

        '''
        for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(image=image,
                                  landmark_list=face_landmarks,
                                  connections=mp_face_mesh.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(image=image,
                                  landmark_list=face_landmarks,
                                  connections=mp_face_mesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(image=image,
                                  landmark_list=face_landmarks,
                                  connections=mp_face_mesh.FACEMESH_IRISES,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
        '''

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()