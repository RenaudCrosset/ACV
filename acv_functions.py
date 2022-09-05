import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

def useMyCam():
    '''
    '''

    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(0)
    while(True):
        _, frame = cap.read()
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)
        cv2.imshow('frame',frame)
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)
        key = cv2.waitKey(5) & 0xFF
        # key = cv2.waitKey(10) 
        if key == 27: #ESC key to stop
            break
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def sidebyside(imageleft, imageright):
    newimheight = max(imageleft.shape[0], imageright.shape[0])
    newimwidth = imageleft.shape[1] + imageright.shape[1]
    
    newim = np.zeros((newimheight, newimwidth, 3), dtype=np.uint8)

    newim[:imageleft.shape[0], :imageleft.shape[1]] = imageleft
    newim[:imageright.shape[0], -imageright.shape[1]:] = imageright

    return newim



'''
*****************MEDIAPIPE NEZ ROUGE***********************
'''

def run_filter_with_mediapipe_model(mediapipe_model, mediapipe_based_filter):
    cap = cv2.VideoCapture(0)
    image = None
    results = None
    with mediapipe_model as model:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue     # If loading a video, use 'break' instead of 'continue'
            # Flip the image horizontally for a later selfie-view display, 
            # and convert the BGR image to RGB :
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = model.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            result_image = mediapipe_based_filter(image, results)
            cv2.imshow('MediaPipe', result_image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()    
    return image, results

def draw_holistic_results(image, results):
    landmarks = results.face_landmarks.landmark
    nose = landmarks[4] # légèrement au-dessus de 1 (sommet du nez)
    # image dimensions
    height, width = image.shape[:2]
    # Center coordinates
    center_coordinates = (int(nose.x*width), int(nose.y*height))   
    # Radius of circle
    radius = int(-400*nose.z+1)
    # Red color in BGR
    color = (0, 0, 255)
    # Line thickness of -1 px
    thickness = -1
    # Using cv2.circle() method
    # Draw a circle of red color of thickness -1 px
    image = cv2.circle(image, center_coordinates, radius, color, thickness)
    return image

def nez_rouge():
    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    Holistic = mp.solutions.holistic.Holistic

    holistic_model = Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    last_image, last_results = run_filter_with_mediapipe_model(mediapipe_model=holistic_model, mediapipe_based_filter=draw_holistic_results)


'''
*****************MEDIAPIPE NEZ ROUGE AVEC REGLAGE********************
'''

def run_filter_with_mediapipe_model_reglable(mediapipe_model, mediapipe_based_filter):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Mon_nez_rouge') 
    cv2.createTrackbar("Nez", "Mon_nez_rouge", 1, 10, on_trackbar)
    image = None
    results = None
    with mediapipe_model as model:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue     # If loading a video, use 'break' instead of 'continue'
            # Flip the image horizontally for a later selfie-view display, 
            # and convert the BGR image to RGB :
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = model.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            result_image = mediapipe_based_filter(image, results)
            cv2.imshow('Mon_nez_rouge', result_image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()    
    return image, results

def on_trackbar(val):
    taille_nez = 0.5 + val/10

def draw_holistic_results_reglable(image, results):
    if results.face_landmarks==None:
        return image
    else:
        landmarks = results.face_landmarks.landmark
        nose = landmarks[4] # légèrement au-dessus de 1 (sommet du nez)
        # image dimensions
        height, width = image.shape[:2]
        # Center coordinates
        center_coordinates = (int(nose.x*width), int(nose.y*height))   
        # Radius of circle
        taille_nez = cv2.getTrackbarPos('Nez','Mon_nez_rouge')
        radius = int(-400*nose.z+1) * taille_nez
        # Red color in BGR
        color = (0, 0, 255)
        # Line thickness of -1 px
        thickness = -1
        # Using cv2.circle() method
        # Draw a circle of red color of thickness -1 px
        image = cv2.circle(image, center_coordinates, radius, color, thickness)
        return image

def nez_rouge_reglable():
    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    Holistic = mp.solutions.holistic.Holistic

    holistic_model = Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    last_image, last_results = run_filter_with_mediapipe_model_reglable(mediapipe_model=holistic_model, mediapipe_based_filter=draw_holistic_results_reglable)


'''
*****************MEDIAPIPE MOUSTACHE***********************
'''

def get_coords(point, image):
    x = int(image.shape[1] * point.x)
    y = int(image.shape[0] * point.y)
    return (x, y)

def add_halfstash(minX, maxX, minY, maxY, stash, image):
    stash = cv2.resize(stash, (maxX - minX, maxY - minY))
    alpha = stash[:, :, 3] / 255.0
    alpha_inv = 1 - alpha
    for c in range(0,3):
        image[minY:maxY, minX:maxX, c] = (alpha * stash[:,:,c] + alpha_inv * image[minY:maxY, minX:maxX, c])

def draw(face, image):
    lip_mid = face[164]
    lip_left = face[322]
    lip_right = face[92]
    lip_upper = face[2]
    lip_lower = face[0]
    x_mid, y_mid = get_coords(lip_mid, image)
    x_left, y_left = get_coords(lip_left, image)
    x_right, y_right = get_coords(lip_right, image)
    _, y_upper = get_coords(lip_upper, image)
    _, y_lower = get_coords(lip_lower, image)
    stash = cv2.imread('data/halfstash.png', cv2.IMREAD_UNCHANGED)
    add_halfstash(x_mid, x_left, y_upper, y_lower, stash, image)
    stash = cv2.flip(stash, 1)
    add_halfstash(x_right, x_mid, y_upper, y_lower, stash, image)

def moustache():
    cap = cv2.VideoCapture(0)
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        face = face_landmarks.landmark
                        draw(face, image)
                cv2.imshow('Face', cv2.flip(image, 1)) #selfie flip
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    cap.release()
    cv2.destroyAllWindows()

'''
*****************MEDIAPIPE GROUCHO MARX***********************
'''

def effect_image_face_2(image, results):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    im_height, im_width, _ = image.shape
    obj = cv2.imread("data/groucho-marx_resized.png", cv2.IMREAD_UNCHANGED)
    # obj = cv2.imread("data/carnaval.png", cv2.IMREAD_UNCHANGED)
    # pts1 = np.float32([[25, 75],[290, 75],[160, 250]]) # groucho-marx_resized.
    pts1 = np.float32([[25, 75],[290, 75],[160, 250]]) # carnaval
    if results.multi_face_landmarks:
        for face_num, face_landmarks in enumerate(results.multi_face_landmarks):
            eye_1 = face_landmarks.landmark[71] # groucho-marx_resized.
            eye_2 = face_landmarks.landmark[301]
            nose = face_landmarks.landmark[164]
            # eye_1 = face_landmarks.landmark[71]
            # eye_2 = face_landmarks.landmark[301]
            # nose = face_landmarks.landmark[164]
            pts2 = np.float32([[eye_1.x*im_width, eye_1.y*im_height],[eye_2.x*im_width, eye_2.y*im_height],[nose.x*im_width, nose.y*im_height]])
            M = cv2.getAffineTransform(pts1,pts2)
            dst = cv2.warpAffine(obj,M,(im_width,im_height))
            color = dst[:, :, :3]
            alpha = dst[:, :, 3:] / 255.  # should be an array between 0 and 1
            image = alpha * color + (1-alpha) * image
            image = image.astype(np.uint8)
            # cv2.imshow('dst', dst)
    return image


def groucho():
    cap = cv2.VideoCapture(0)
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break



                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                image.flags.writeable = True


                image = effect_image_face_2(image, results)

                cv2.imshow('Face', cv2.flip(image, 1)) #selfie flip
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    cap.release()
    cv2.destroyAllWindows()



'''
*****************MEDIAPIPE FINGER COUNTER***********************
cf hand_tracking_module.py
'''


'''
*****************MEDIAPIPE FACE SWAP***********************

cf face_landmarks_extractor.py
cf face_warper.py

'''





