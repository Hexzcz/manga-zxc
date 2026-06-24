from src.translation.deepseek import DeepseekApi
from src.data.core import TranslationPage


class TranslationManager:
    def __init__(
        self,
        api_key: str,
        api_model: str
    ):
        self.api_key = api_key
        self.api_model = api_model

        self._deepseek_translator = None

        self._load_translator()
    

    def _load_translator(self):
        self._deepseek_translator = DeepseekApi(
            api_key=self.api_key,
            model=self.api_model
        )
    

    def start(self, pages: list[TranslationPage]) -> list[TranslationPage]:
        translation_data = {}

        # Build payload matching the prompt's expected format:
        # { page_idx: { bubble_idx: { cluster_idx: { jp_text: ..., tr_text: "" } } } }
        for page in pages:
            bubble_data = {}
            for bubble_idx, bubble in enumerate(page.speech_bubbles):
                cluster_data = {
                    cluster_idx: {"jp_text": cluster.jp_text or "", "tr_text": ""}
                    for cluster_idx, cluster in enumerate(bubble.text_clusters)
                }
                bubble_data[bubble_idx] = cluster_data
            translation_data[page.index] = bubble_data

        translated = self._deepseek_translator.translate(translation_data)
        print(translated)

        for page in pages:
            page_trans = translated.get(str(page.index), {})
            for b_idx, bubble in enumerate(page.speech_bubbles):
                cluster_trans = page_trans.get(str(b_idx), {})
                for c_idx, cluster in enumerate(bubble.text_clusters):
                    raw = cluster_trans.get(str(c_idx), "")
                    if isinstance(raw, dict):
                        # Expected: LLM returned {jp_text: ..., tr_text: "..."}
                        t_str = raw.get("tr_text") or ""
                        j_str = raw.get("jp_text") or ""
                        cluster.tr_text = t_str.strip() or j_str.strip()
                    elif isinstance(raw, str):
                        # Fallback: LLM returned a bare translated string
                        cluster.tr_text = raw.strip()
                    else:
                        cluster.tr_text = ""

        return pages