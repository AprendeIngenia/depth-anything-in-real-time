import cv2
import numpy as np
import onnxruntime as ort

from depth_anything.util.transform import load_frame


def load_depth_model(model: str) -> ort.InferenceSession:
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    return session


def preprocess_depth_output(depth_output: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    depth = cv2.resize(depth_output[0, 0], (orig_w, orig_h))
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    return depth


def visualize_depth_color(depth: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)


class DepthAnythingInference:
    def __init__(self, model_path: str, color: bool = False):
        self.model_path = model_path
        self.color = color

    def frame_inference(self, frame: np.ndarray) -> np.ndarray:
        image, (orig_h, orig_w) = load_frame(frame)
        session = load_depth_model(self.model_path)
        depth_output = session.run(None, {"image": image})[0]
        depth = preprocess_depth_output(depth_output, orig_h, orig_w)

        if self.color:
            depth = visualize_depth_color(depth)

        return depth
