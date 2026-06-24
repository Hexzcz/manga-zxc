import time
from multiprocessing import Queue
from src.data.core import TranslationPage, strip_ipc_crop_payload
from src.recognition.manager import RecognitionManager
from src.pipeline_timing import append_stage_timing

class RecognitionWorker:
    def __init__(
        self,
        in_queue: Queue,
        out_queue: Queue,
        shared_state,
        state_lock,
        timing_log=None,
    ):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.timing_log = timing_log
        self._recognition_manager = None
        self._init_manager()
        with self.state_lock:
            self.shared_state["ocr_ready"] = True
    
    def _init_manager(self):
        self._recognition_manager = RecognitionManager()
    
    def run(self):
        while True:
            page_data: TranslationPage = self.in_queue.get()
            if page_data is None:
                self.out_queue.put(None)
                break
            
            # Mark page as running
            with self.state_lock:
                ps = self.shared_state["process_status"]
                rec_data = ps["recognition"]
                rec_data["status"] = "running"
                rec_data["current_file"] = page_data.file_path
                ps["recognition"] = rec_data
                self.shared_state["process_status"] = ps
            
            t0 = time.perf_counter()
            try:
                # ---- ACTUAL RECOGNITION ----
                page_data = self._recognition_manager.start(page_data)
            except Exception as e:
                print(f"Error in RecognitionWorker: {e}")
            finally:
                append_stage_timing(self.timing_log, "recognition", page_data.index, t0)
            
            # Increment completed_pages, mark finished when all done
            with self.state_lock:
                ps = self.shared_state["process_status"]
                det_data = ps["detection"]
                total_pages = det_data.get("total_pages", 0)
                
                rec_data = ps["recognition"]
                rec_data["completed_pages"] = rec_data.get("completed_pages", 0) + 1
                
                if rec_data["completed_pages"] == total_pages:
                    rec_data["status"] = "finished"
                    rec_data["current_file"] = None
                
                ps["recognition"] = rec_data
                self.shared_state["process_status"] = ps

            # Flaw 1 optimization: shrink Queue pickle (re-cut crops in inpainting after load).
            strip_ipc_crop_payload(page_data)
            self.out_queue.put(page_data)