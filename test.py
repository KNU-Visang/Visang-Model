"""for test.py
pip install opencv-python
pip install dlib
pip install numpy
pip install imutils #for face_utils module
pip install tensorflow
pip install keras
"""
import cv2, dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model

def crop_eye(img, eye_points): #눈 검출
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)

    eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect

def eye_blink_detector(img_path): #눈 깜빡임 확인
    img_ori = cv2.imread(img_path)
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)

        eye_img_l, _ = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, _ = crop_eye(gray, eye_points=shapes[42:48])

        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)

        eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(float) / 255.
        eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(float) / 255.

        pred_l = model.predict(eye_input_l, verbose=0)[0][0]
        pred_r = model.predict(eye_input_r, verbose=0)[0][0]


        # 터미널에 출력
        state_l = 'Open' if pred_l > 0.1 else 'Closed'
        state_r = 'Open' if pred_r > 0.1 else 'Closed'

        print(f'Left eye: {state_l} ({pred_l:.2f}), Right eye: {state_r} ({pred_r:.2f})') 
        # 일단 이거 print로 해놨는데, 이후에 개발 상황보고 리턴값 수정해주면 됨

IMG_SIZE = (34, 26) #

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat') #눈 검출을 위한 dlib 모델

model = load_model('models/eye_blink_detector_model.keras')
# model.summary()

eye_blink_detector("images/test1.png")