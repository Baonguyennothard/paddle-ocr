import cv2
from config.config import OCRConfig
from tools.predict_rec import TextRecognizer


args = OCRConfig()
text_recognizer = TextRecognizer(args)
rec_res, _ = text_recognizer([cv2.imread('23893m.png')])
print(rec_res)