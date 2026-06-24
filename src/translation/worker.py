import time
from multiprocessing import Queue
from src.data.core import TranslationPage
from src.translation.manager import TranslationManager
from src.pipeline_timing import append_stage_timing

class TranslationWorker:
    def __init__(
        self,
        api_key: str,
        api_model: str,
        in_queue: Queue,
        out_queue: Queue,
        shared_state,
        state_lock,
        timing_log=None,
    ):
        self.api_key = api_key
        self.api_model = api_model
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.timing_log = timing_log
        self._translation_manager = None
        self._init_manager()
    
    def _init_manager(self):
        self._translation_manager = TranslationManager(
            api_key=self.api_key,
            api_model=self.api_model
        )
    
    def run(self):
        pages: list[TranslationPage] = []
        while True:
            page_data: TranslationPage = self.in_queue.get()
            if page_data is None:
                break
            
            pages.append(page_data)
            
            with self.state_lock:
                det_data = self.shared_state["process_status"]["detection"]
                total_pages = det_data.get("total_pages", 0)
            
            # Start translation batch once all pages collected
            if len(pages) == total_pages and total_pages > 0:
                try:
                    with self.state_lock:
                        ps = self.shared_state["process_status"]
                        tra_data = ps["translation"]
                        tra_data["status"] = "running"
                        tra_data["current_file"] = f"Translating {len(pages)} pages"
                        tra_data["completed_pages"] = 0
                        ps["translation"] = tra_data
                        self.shared_state["process_status"] = ps
                    
                    # ---- ACTUAL TRANSLATION ----
                    t0 = time.perf_counter()
                    try:
                        result = self._translation_manager.start(pages)
                    finally:
                        append_stage_timing(
                            self.timing_log,
                            "translation_batch",
                            None,
                            t0,
                            pages=len(pages),
                        )
                    
                    with self.state_lock:
                        ps = self.shared_state["process_status"]
                        tra_data = ps["translation"]
                        tra_data["status"] = "finished"
                        tra_data["current_file"] = None
                        tra_data["completed_pages"] = total_pages
                        ps["translation"] = tra_data
                        self.shared_state["process_status"] = ps
                    
                    for page in result:
                        self.out_queue.put(page)
                except Exception as e:
                    print(f"Error in TranslationWorker: {e}")
                    with self.state_lock:
                        ps = self.shared_state["process_status"]
                        tra_data = ps["translation"]
                        tra_data["status"] = "failed"
                        tra_data["current_file"] = None
                        ps["translation"] = tra_data
                        self.shared_state["process_status"] = ps
                    # Push unprocessed pages downstream on error
                    for page in pages:
                        self.out_queue.put(page)
                finally:
                    pages.clear()
            
        self.out_queue.put(None)