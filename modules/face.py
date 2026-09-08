import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceAnalyzer:
    def __init__(self, model_name="buffalo_l", providers=None):
        providers = providers or ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        return self.app.get(image)

    def embedding(self, image_path):
        faces = self.detect(image_path)
        if not faces:
            return None
        face = max(faces, key=lambda x: x.det_score)
        embedding = face.embedding.astype(np.float32)
        return embedding / np.linalg.norm(embedding)

    def verify(self, image1, image2, threshold=0.40):
        emb1 = self.embedding(image1)
        emb2 = self.embedding(image2)

        if emb1 is None or emb2 is None:
            return {"match": False, "similarity": 0.0, "reason": "Face not detected"}

        similarity = float(np.dot(emb1, emb2))
        return {"match": similarity >= threshold, "similarity": similarity}


def detect_faces(image_path):
    return FaceAnalyzer().detect(image_path)


def verify_faces(image1, image2, threshold=0.40):
    return FaceAnalyzer().verify(image1, image2, threshold)