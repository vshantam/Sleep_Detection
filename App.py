#importing Libraries

from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2

#creating class for eye_aspect_ratio
class EAR(object):

    @classmethod
    def eye_aspect_ratio(self,eye):
    	A = distance.euclidean(eye[1], eye[5])
    	B = distance.euclidean(eye[2], eye[4])
    	C = distance.euclidean(eye[0], eye[3])
    	ear = (A + B) / (2.0 * C)
    	return ear

#creating class for sleep detection
class SLPD(object):

    #initializing values
    @classmethod
    def __init__(self):

        self.thresh = 0.25
        self.frame_check = 10
        self.detect = dlib.get_frontal_face_detector()
        self.predict = dlib.shape_predictor("Model/shape_predictor_68_face_landmarks.dat")

    #starting the operation
    @classmethod
    def start(self):

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        self.cap=cv2.VideoCapture(0)
        self.flag=0
        EAR_Obj = EAR()
        if self.cap.isOpened():
            while True:
            	ret, frame=self.cap.read()
            	frame = imutils.resize(frame, width=450)
            	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            	subjects = self.detect(gray, 0)
            	for subject in subjects:
            		shape = self.predict(gray, subject)
            		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
            		self.leftEye = shape[self.lStart:self.lEnd]
            		self.rightEye = shape[self.rStart:self.rEnd]
            		self.leftEAR = EAR_Obj.eye_aspect_ratio(self.leftEye)
            		self.rightEAR = EAR_Obj.eye_aspect_ratio(self.rightEye)
            		ear = (self.leftEAR + self.rightEAR) / 2.0
            		leftEyeHull = cv2.convexHull(self.leftEye)
            		rightEyeHull = cv2.convexHull(self.rightEye)
            		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            		if ear < self.thresh:
            			self.flag += 1
            			cv2.putText(frame, str(self.flag), (0, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            			if self.flag >= self.frame_check:
            				cv2.putText(frame, "****************ALERT!****************", (10, 30),
            					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            				cv2.putText(frame, "****************ALERT!****************", (10,325),
            					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            				#print ("Drowsy")
            		else:
            			self.flag = 0
            	cv2.imshow("Frame", frame)
            	key = cv2.waitKey(1) & 0xFF
            	if key == ord("q"):
                    cv2.destroyAllWindows()
                    self.cap.release()

if __name__ == "__main__":

    Start_obj = SLPD()
    try:
        Start_obj.start()
    except Exception as e:
        print("Error:Occured : {}".format(str(e)))
