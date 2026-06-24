import onnxruntime

from src.data.core import TranslationPage
from src.inpainting.migan_inpainter import MiganInpainter
from src.inpainting.text_masking import batch_apply_text_mask


class InpaintingManager:
    def __init__(
        self,
        migan_model_file_path: str
    ):
        self.migan_model_file_path = migan_model_file_path

        self._inpainting_model = None
        self._migan_inpainter = None
        
        self._load_migan_model()
        self._init_class()

    
    def _load_migan_model(self):
        self._inpainting_model = onnxruntime.InferenceSession(self.migan_model_file_path)
    
    
    def _init_class(self):
        self._migan_inpainter = MiganInpainter(
            model=self._inpainting_model
        )


    def start(self, page_data: TranslationPage) -> TranslationPage:
        image_pil = page_data.image_pil
        blank_canvas = page_data.blank_canvas
        text_clusters_data = [cluster for bubble in page_data.speech_bubbles for cluster in bubble.text_clusters]
        image_mask = batch_apply_text_mask("migan", blank_canvas, text_clusters_data)
        
        page_data.cleaned_image = self._migan_inpainter.inpaint(image_pil, image_mask)
        return page_data