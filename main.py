import cv2
from depth_anything.inference import DepthAnythingInference


class IntelligentDodgeRobot:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.depth_anything = DepthAnythingInference(model_path='depth_anything/models/depth_anything_vits14.onnx',
                                                     color=True)

    def frame_processing(self):
        while True:
            t = cv2.waitKey(5)
            ret, frame = self.cap.read()
            img_depth = self.depth_anything.frame_inference(frame)

            cv2.imshow('depth anything', img_depth)
            cv2.imshow('frame in real time', frame)
            if t == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


robot_intelligent = IntelligentDodgeRobot()
robot_intelligent.frame_processing()
