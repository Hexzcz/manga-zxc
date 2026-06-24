from ultralytics import YOLO
import cv2

from src.detection.speech_bubble import SpeechBubbleDetector
from src.detection.text_cluster import TextClusterDetector
from src.data.core import TranslationPage


class DetectionManager:
    def __init__(
        self,
        speech_bubble_model_file_path: str,
        text_cluster_model_file_path: str
    ):
        self._speech_bubble_model_file_path = speech_bubble_model_file_path
        self._text_cluster_model_file_path = text_cluster_model_file_path

        self._speech_bubble_model = None
        self._text_cluster_model = None
        self.var = 0

        self._load_yolo_model()
        self._init_class()
    

    def _load_yolo_model(self):
        try:
            self._speech_bubble_model = YOLO(self._speech_bubble_model_file_path)
            self._text_cluster_model = YOLO(self._text_cluster_model_file_path)
        except Exception as e:
            print(f"Loading YOLO model failed: {e}")
    

    def _init_class(self):
        self._speech_bubble_detector = SpeechBubbleDetector(
            model=self._speech_bubble_model
        )
        self._text_cluster_detector = TextClusterDetector(
            model=self._text_cluster_model
        )
    

    def start(self, page_data: TranslationPage) -> TranslationPage:
        speech_bubbles = self._speech_bubble_detector.detect(page_data.image_cv2)

        if speech_bubbles:
            speech_bubbles = self._text_cluster_detector.detect(speech_bubbles)
        
        self._visualize_detection(page_data.image_cv2, speech_bubbles)
        
        page_data.speech_bubbles = speech_bubbles
        return page_data
    

    def _visualize_detection(self, image_cv2, speech_bubbles_data):
        image_copy = image_cv2.copy()

        for speech_bubble_data in speech_bubbles_data:
            SBx_min, SBy_min, SBx_max, SBy_max = speech_bubble_data.position
            cv2.rectangle(image_copy, (SBx_min, SBy_min), (SBx_max, SBy_max), (255, 0, 0), 4)
        
            for text_cluster_data in speech_bubble_data.text_clusters:
                TCx_min, TCy_min, TCx_max, TCy_max = text_cluster_data.position
                cv2.rectangle(image_copy, (TCx_min, TCy_min), (TCx_max, TCy_max), (0, 255, 0), 4)
        
        cv2.imwrite(f"backend/pages/output/visualized_detection{self.var}.png", image_copy)
        self.var += 1


    
        

