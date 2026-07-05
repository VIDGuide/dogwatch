"""detector.py — thin wrapper around pycoral, filtered to the 'dog' class."""
import cv2
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect


class DogDetector:
    def __init__(self, model_path, labels_path, score_threshold=0.4,
                 target_label="dog"):
        self.interp = make_interpreter(model_path)
        self.interp.allocate_tensors()
        self.score_threshold = score_threshold
        self.labels = self._load_labels(labels_path)
        # COCO label files vary ("dog" may be id 17 or 18) — resolve by name.
        self.target_ids = {i for i, n in self.labels.items()
                           if n.lower() == target_label.lower()}
        if not self.target_ids:
            raise ValueError(f"'{target_label}' not found in {labels_path}")

    @staticmethod
    def _load_labels(path):
        labels = {}
        with open(path) as f:
            for idx, line in enumerate(f):
                name = line.strip()
                if name:
                    labels[idx] = name
        return labels

    def detect(self, frame):
        """Return [{'bbox': (x0,y0,x1,y1) in pixels, 'score': float}, ...]."""
        h, w = frame.shape[:2]
        _, scale = common.set_resized_input(
            self.interp, (w, h), lambda size: cv2.resize(frame, size))
        self.interp.invoke()
        out = []
        for obj in detect.get_objects(self.interp, self.score_threshold, scale):
            if obj.id in self.target_ids:
                b = obj.bbox
                out.append({"bbox": (b.xmin, b.ymin, b.xmax, b.ymax),
                            "score": obj.score})
        return out
