from dataclasses import dataclass, field
import numpy as np
from PIL import Image
from typing import Optional
from io import BytesIO

@dataclass
class TextCluster:
    position: tuple[int, int, int, int]
    confidence: float
    bounding_image: np.ndarray | None = None
    jp_text: str | None = None
    tr_text: str | None = None
    area: float | None = None


@dataclass
class SpeechBubble:
    position: tuple[int, int, int, int]
    confidence: float
    class_name: str
    bounding_image: np.ndarray | None = None
    text_clusters: list[TextCluster] = field(default_factory=list)


@dataclass
class TranslationPage:
    index: int
    file_path: str
    image_cv2: Optional[np.ndarray] = None
    image_pil: Optional[Image.Image] = None
    blank_canvas: Optional[np.ndarray] = None
    cleaned_image: Optional[Image.Image] = None
    speech_bubbles: list[SpeechBubble] = field(default_factory=list)

    def load(self):
        """Decode the file from disk into RAM only when actively needed by a worker."""
        if self.image_cv2 is None and self.file_path:
            self.image_pil = Image.open(self.file_path).convert("RGB")
            self.image_cv2 = cv2.cvtColor(np.array(self.image_pil), cv2.COLOR_RGB2BGR)
            self.blank_canvas = np.ones_like(self.image_cv2, dtype=np.uint8) * 255

    def unload(self):
        """Immediately clear the 60MB+ of arrays to prevent multiprocessing queue backlog crashes."""
        self.image_cv2 = None
        self.image_pil = None
        self.blank_canvas = None
        self.cleaned_image = None  # Free inpainting modifications after disk save
@dataclass
class TranslationBatch:
    pages: list[TranslationPage] = field(default_factory=list)


import os
import cv2


def load_pages_from_bytes(idx: int, filename: str, image_bytes: bytes) -> TranslationPage:
    if idx != -1:
        image_pil_load = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        image_cv2_load = cv2.cvtColor(np.array(image_pil_load), cv2.COLOR_RGB2BGR)
        
        blank_canvas_load = np.ones_like(image_cv2_load, dtype=np.uint8) * 255
    else:
        image_pil_load = None
        image_cv2_load = None
        blank_canvas_load = None
    
    return TranslationPage(
        index=idx,
        file_path=filename,
        image_cv2=image_cv2_load,
        image_pil=image_pil_load,
        blank_canvas=blank_canvas_load
    )


def to_cv2(image_path: str) -> np.ndarray:
    return cv2.imread(image_path)


def to_pil(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def cv2_to_pil(image_cv2: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))


def blank_canvas(image_cv2: np.ndarray) -> np.ndarray:
    return np.ones_like(image_cv2, dtype=np.uint8) * 255


def strip_ipc_crop_payload(page: TranslationPage) -> None:
    """
    Flaw 1 optimization: clear bubble/cluster numpy crops before multiprocessing.Queue
    transfer so pickling does not ship large ndarray payloads (positions + text remain).
    """
    for sb in page.speech_bubbles or []:
        sb.bounding_image = None
        for tc in sb.text_clusters or []:
            tc.bounding_image = None


def attach_text_cluster_crops_from_page(page: TranslationPage) -> None:
    """
    Flaw 1 optimization: after load(), rebuild TextCluster.bounding_image slices from the
    full page using absolute positions (same pixels as detection, without IPC copies).
    """
    img = page.image_cv2
    if img is None:
        return
    h, w = img.shape[:2]
    for sb in page.speech_bubbles or []:
        for tc in sb.text_clusters or []:
            x1, y1, x2, y2 = tc.position
            xa, xb = sorted((int(x1), int(x2)))
            ya, yb = sorted((int(y1), int(y2)))
            xa = max(0, min(xa, w))
            xb = max(0, min(xb, w))
            ya = max(0, min(ya, h))
            yb = max(0, min(yb, h))
            if xb > xa and yb > ya:
                tc.bounding_image = img[ya:yb, xa:xb].copy()