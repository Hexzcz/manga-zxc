import time
from multiprocessing import Queue
from pathlib import Path
from src.data.core import TranslationPage
from PIL import Image
import json

from src.pipeline_timing import append_stage_timing


class TypesettingWorker:
    """
    Typesetting worker — saves cleaned images and metadata JSON for
    frontend-based text overlay rendering via Fabric.js.
    """

    def __init__(
        self,
        in_queue: Queue,
        shared_state,
        state_lock,
        timing_log=None,
    ):
        self.in_queue = in_queue
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.timing_log = timing_log

    def _get_output_dir(self) -> Path:
        try:
            out = self.shared_state.get("output_dir", "")
            p = Path(out)
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            fallback = Path(__file__).resolve().parents[3] / "pages" / "output" / "unknown"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    def _save_metadata(self, data: TranslationPage, output_dir: Path):
        """
        Save per-page metadata JSON containing bubble/cluster positions
        and translated text for the frontend Fabric.js overlay.
        Coordinates are stored as percentages of image dimensions.
        """
        meta_path = output_dir / f"{data.index:04d}_meta.json"

        img_w, img_h = 1200, 1800
        try:
            with Image.open(data.file_path) as im:
                img_w, img_h = im.size
        except Exception:
            pass

        bubbles = []
        for sb_idx, sb in enumerate(data.speech_bubbles or []):
            sb_x1, sb_y1, sb_x2, sb_y2 = sb.position

            clusters = []
            for tc_idx, tc in enumerate(sb.text_clusters or []):
                tc_x1, tc_y1, tc_x2, tc_y2 = tc.position

                clusters.append({
                    "id": f"p{data.index}_b{sb_idx}_c{tc_idx}",
                    "x_pct": round((tc_x1 / img_w) * 100, 3),
                    "y_pct": round((tc_y1 / img_h) * 100, 3),
                    "w_pct": round(((tc_x2 - tc_x1) / img_w) * 100, 3),
                    "h_pct": round(((tc_y2 - tc_y1) / img_h) * 100, 3),
                    "tr_text": tc.tr_text or "",
                    "jp_text": tc.jp_text or "",
                })

            bubbles.append({
                "id": f"p{data.index}_b{sb_idx}",
                "class_name": sb.class_name,
                "x_pct": round((sb_x1 / img_w) * 100, 3),
                "y_pct": round((sb_y1 / img_h) * 100, 3),
                "w_pct": round(((sb_x2 - sb_x1) / img_w) * 100, 3),
                "h_pct": round(((sb_y2 - sb_y1) / img_h) * 100, 3),
                "clusters": clusters,
            })

        metadata = {
            "page_index": data.index,
            "file_path": data.file_path,
            "image_width": img_w,
            "image_height": img_h,
            "bubbles": bubbles,
        }

        with open(str(meta_path), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"TypesettingWorker: Saved metadata {data.index} -> {meta_path}")

    def run(self):
        while True:
            data: TranslationPage = self.in_queue.get()

            if data is None:
                break

            t0 = time.perf_counter()
            try:
                output_dir = self._get_output_dir()
                self._save_metadata(data, output_dir)

                # Update shared state
                with self.state_lock:
                    ps = self.shared_state["process_status"]
                    ts_data = ps["typesetting"]
                    det_data = ps["detection"]
                    total_pages = det_data.get("total_pages", 0)

                    if ts_data["status"] == "waiting":
                        ts_data["status"] = "running"

                    ts_data["completed_pages"] = ts_data.get("completed_pages", 0) + 1
                    ts_data["current_file"] = data.file_path

                    if total_pages > 0 and ts_data["completed_pages"] >= total_pages:
                        ts_data["status"] = "finished"
                        ts_data["current_file"] = None

                    ps["typesetting"] = ts_data
                    self.shared_state["process_status"] = ps
            finally:
                append_stage_timing(self.timing_log, "typesetting", data.index, t0)
