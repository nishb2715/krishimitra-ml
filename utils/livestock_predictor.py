import tensorflow as tf
import numpy as np
from PIL import Image
import json, io, os

class LivestockPredictor:
    def __init__(self):
        base = os.path.join(os.path.dirname(__file__), '..', 'models')
        self.interpreter = tf.lite.Interpreter(
            model_path=os.path.join(base, 'livestock_model.tflite')
        )
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        with open(os.path.join(base, 'livestock_labels.json'), encoding='utf-8') as f:
            self.labels = json.load(f)
        print(f"LivestockPredictor loaded — {len(self.labels)} classes")

    def predict(self, image_bytes: bytes) -> dict:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], arr)
        self.interpreter.invoke()
        preds      = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        top_idx    = str(np.argmax(preds))
        confidence = float(np.max(preds))
        info       = self.labels[top_idx]
        return {
            "status":           "success",
            "predicted_class":  info["class_name"],
            "odia_name":        info["odia_name"],
            "confidence":       round(confidence * 100, 1),
            "severity":         info["severity"],
            "advice_odia":      info["advice_odia"],
            "home_remedy":      info.get("home_remedy", ""),
            "see_vet_urgently": info["see_vet_urgently"],
            "low_confidence":   confidence < 0.60
        }

livestock_predictor = LivestockPredictor()