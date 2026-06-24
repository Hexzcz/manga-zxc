from manga_ocr import MangaOcr

from src.recognition.ocr import TextRecognition
from src.data.core import TranslationPage


class RecognitionManager:
    def __init__(self):
        self._mangaocr_model = None

        self._load_mangaocr_model( )
        self._init_class()
    

    def _load_mangaocr_model(self):
        try:
            self._mangaocr_model = MangaOcr()
        except Exception as e:
            print(f"Loading MangaOcr model failed: {e}")
    

    def _init_class(self):
        self._text_recognition_process = TextRecognition(
            manga_ocr=self._mangaocr_model
        )
    

    def start(self, page_data: TranslationPage) -> TranslationPage:
        speech_bubbles_data = page_data.speech_bubbles

        recognized_data = self._text_recognition_process.text_recognition(speech_bubbles_data)

        if recognized_data:
            page_data.speech_bubbles = recognized_data
        
        return page_data