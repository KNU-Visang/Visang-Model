import platform, os
import cv2, dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model
class EyeBlinkDetector():
    FILE_PATH = os.path.dirname(__file__) + "/" if platform.system() == "Linux" else ""

    IMG_SIZE = (34, 26)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FILE_PATH + 'models/shape_predictor_68_face_landmarks.dat')

    model = load_model(FILE_PATH + 'models/eye_blink_detector_model.keras')
    
    def crop_eye(self, img, eye_points):
        x1, y1 = np.amin(eye_points, axis=0)
        x2, y2 = np.amax(eye_points, axis=0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        w = (x2 - x1) * 1.2
        h = w * EyeBlinkDetector.IMG_SIZE[1] / EyeBlinkDetector.IMG_SIZE[0]

        margin_x, margin_y = w / 2, h / 2

        min_x, min_y = int(cx - margin_x), int(cy - margin_y)
        max_x, max_y = int(cx + margin_x), int(cy + margin_y)

        eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)

        eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

        return eye_img, eye_rect

    def eye_blink_detector(self, img_path):
        img_ori = cv2.imread(img_path)
        gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

        faces = EyeBlinkDetector.detector(gray)
        
        ret = []

        for face in faces:
            shapes = EyeBlinkDetector.predictor(gray, face)
            shapes = face_utils.shape_to_np(shapes)

            eye_img_l, _ = self.crop_eye(gray, eye_points=shapes[36:42])
            eye_img_r, _ = self.crop_eye(gray, eye_points=shapes[42:48])

            eye_img_l = cv2.resize(eye_img_l, dsize=EyeBlinkDetector.IMG_SIZE)
            eye_img_r = cv2.resize(eye_img_r, dsize=EyeBlinkDetector.IMG_SIZE)
            eye_img_r = cv2.flip(eye_img_r, flipCode=1)

            eye_input_l = eye_img_l.copy().reshape((1, EyeBlinkDetector.IMG_SIZE[1], EyeBlinkDetector.IMG_SIZE[0], 1)).astype(float) / 255.
            eye_input_r = eye_img_r.copy().reshape((1, EyeBlinkDetector.IMG_SIZE[1], EyeBlinkDetector.IMG_SIZE[0], 1)).astype(float) / 255.

            pred_l = EyeBlinkDetector.model.predict(eye_input_l, verbose=0)[0][0]
            pred_r = EyeBlinkDetector.model.predict(eye_input_r, verbose=0)[0][0]


            state_l = 'Open' if pred_l > 0.1 else 'Closed'
            state_r = 'Open' if pred_r > 0.1 else 'Closed'
            
            ret.append((pred_l <= 0.1, pred_r <= 0.1))

            print(f'Left eye: {state_l} ({pred_l:.2f}), Right eye: {state_r} ({pred_r:.2f})')
        
        return ret
            
            

if __name__ == "__main__":
    eyeBlinkDetector = EyeBlinkDetector()
    eyeBlinkDetector.eye_blink_detector(EyeBlinkDetector.FILE_PATH + "images/test1.png")
    eyeBlinkDetector.eye_blink_detector(EyeBlinkDetector.FILE_PATH + "images/test2.png")
