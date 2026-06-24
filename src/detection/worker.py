import time
from multiprocessing import Queue
from src.data.core import TranslationPage
from src.detection.manager import DetectionManager
from src.pipeline_timing import append_stage_timing


class DetectionWorker:
    def __init__(
        self,
        speech_bubble_model_file_path: str,
        text_cluster_model_file_path: str,
        in_queue: Queue,
        out_queue: Queue,
        shared_state,
        state_lock,
        timing_log=None,
    ):
        self.speech_bubble_model_file_path = speech_bubble_model_file_path
        self.text_cluster_model_file_path = text_cluster_model_file_path
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.timing_log = timing_log

        self._detection_manager = None
        self._init_manager()
    

    def _init_manager(self):
        self._detection_manager = DetectionManager(
            speech_bubble_model_file_path=self.speech_bubble_model_file_path,
            text_cluster_model_file_path=self.text_cluster_model_file_path
        )
    

    def run(self):
        while True:
            page_data: TranslationPage = self.in_queue.get()
            if page_data is None:
                self.out_queue.put(None)
                break

            # Mark page as "running" in state
            with self.state_lock:
                ps = self.shared_state["process_status"]
                det_data = ps["detection"]
                
                det_data["status"] = "running"
                det_data["current_index"] = page_data.index
                
                files_list = list(det_data["files"])
                files_list.append({
                    "file_name": page_data.file_path,
                    "index": page_data.index,
                    "status": "waiting"
                })
                det_data["files"] = files_list
                
                ps["detection"] = det_data
                self.shared_state["process_status"] = ps
                print(f"DetectionWorker: Processing page {page_data.index}")

            # ---- ACTUAL DETECTION ----
            t0 = time.perf_counter()
            try:
                page_data.load()
                page_data = self._detection_manager.start(page_data)
                
                final_status = (
                    "success"
                    if page_data.speech_bubbles
                    else "empty"
                )
            except Exception as e:
                print(f"Error in DetectionWorker: {e}")
                final_status = "failed"
            finally:
                append_stage_timing(self.timing_log, "detection", page_data.index, t0)

            # Update page result + increment completed_pages
            with self.state_lock:
                ps = self.shared_state["process_status"]
                det_data = ps["detection"]
                files_list = list(det_data["files"])
                
                if files_list:
                    files_list[-1]["status"] = final_status
                
                det_data["files"] = files_list
                det_data["completed_pages"] = det_data.get("completed_pages", 0) + 1
    
                total_pages = det_data.get("total_pages", 0)
                if det_data["completed_pages"] == total_pages:
                    det_data["status"] = "finished"
                
                ps["detection"] = det_data
                self.shared_state["process_status"] = ps
                print(f"DetectionWorker: Finished page {page_data.index} with status {final_status}")
            
            # Immediately free RAM before queuing for recognition!
            page_data.unload()
            self.out_queue.put(page_data)