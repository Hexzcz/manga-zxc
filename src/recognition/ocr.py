from manga_ocr import MangaOcr

from src.data.core import cv2_to_pil
from src.data.core import SpeechBubble, TextCluster


class TextRecognition:
    def __init__(self, manga_ocr: MangaOcr):
        self.manga_ocr = manga_ocr
    

    def _text_recognition_single(self, text_cluster_data: TextCluster) -> str:
        text_cluster_image = text_cluster_data.bounding_image
        text_cluster_image_pil = cv2_to_pil(text_cluster_image)

        ocr_results = self.manga_ocr(text_cluster_image_pil)
        return ocr_results

    
    def _text_recognition_batch(self, speech_bubbles_data: list[SpeechBubble]) -> list[SpeechBubble]:
        new_speech_bubbles = []
        
        for speech_bubble_data in speech_bubbles_data:
            new_text_clusters = []

            if speech_bubble_data.bounding_image is None:
                continue
            
            text_clusters_data = speech_bubble_data.text_clusters
            for text_cluster_data in text_clusters_data:
                if text_cluster_data.bounding_image is None:
                    continue
                jp_text = self._text_recognition_single(text_cluster_data)
                new_text_clusters.append(TextCluster(
                    position=text_cluster_data.position,
                    confidence=text_cluster_data.confidence,
                    bounding_image=text_cluster_data.bounding_image,
                    jp_text=jp_text,
                    tr_text=None
                ))
            
            new_sb = SpeechBubble(
                position=speech_bubble_data.position,
                confidence=speech_bubble_data.confidence,
                class_name=speech_bubble_data.class_name,
                bounding_image=speech_bubble_data.bounding_image,
                text_clusters=new_text_clusters
            )
            new_speech_bubbles.append(new_sb)
        
        return new_speech_bubbles
    

    def text_recognition(self, data: TextCluster | list[SpeechBubble]) -> str | list[SpeechBubble]:
        if isinstance(data, TextCluster):
            return self._text_recognition_single(data)
        elif isinstance(data, list):
            return self._text_recognition_batch(data)