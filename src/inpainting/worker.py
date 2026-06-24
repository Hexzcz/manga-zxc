import time
from multiprocessing import Queue
from src.data.core import attach_text_cluster_crops_from_page, strip_ipc_crop_payload
from src.inpainting.manager import InpaintingManager
from src.pipeline_timing import append_stage_timing

class InpaintingWorker:
    def __init__(
        self,
        migan_model_file_path: str,
        in_queue: Queue,
        out_queue: Queue,
        shared_state,
        state_lock,
        timing_log=None,
    ):
        self.migan_model_file_path = migan_model_file_path
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.timing_log = timing_log

        self._inpainting_manager = None
        self._init_manager()
    
    def _init_manager(self):
        self._inpainting_manager = InpaintingManager(
            migan_model_file_path=self.migan_model_file_path
        )
    
    def run(self):
        while True:
            page_data = self.in_queue.get()
            if page_data is None:
                self.out_queue.put(None)
                break
            
            # Mark page as running
            with self.state_lock:
                ps = self.shared_state["process_status"]
                inp_data = ps["inpainting"]
                inp_data["status"] = "running"
                inp_data["current_file"] = page_data.file_path
                ps["inpainting"] = inp_data
                self.shared_state["process_status"] = ps
            
            try:
                # ---- ACTUAL INPAINTING ----
                page_data.load()
                # Flaw 1 optimization: cluster crops were stripped before Queue; restore from full page.
                attach_text_cluster_crops_from_page(page_data)
                t0 = time.perf_counter()
                try:
                    page_data = self._inpainting_manager.start(page_data)
                finally:
                    append_stage_timing(self.timing_log, "inpainting", page_data.index, t0)

                # Save the cleaned image over the generic copy if inpainting succeeded
                if page_data.cleaned_image:
                    from pathlib import Path
                    out_path = Path(page_data.file_path).with_name(f"{page_data.index:04d}.png")
                    page_data.cleaned_image.save(str(out_path), "PNG")
                    print(f"InpaintingWorker: Saved cleaned page -> {out_path}")

            except Exception as e:
                print(f"Error in InpaintingWorker: {e}")
                
            # Increment completed_pages, mark finished when all done
            with self.state_lock:
                ps = self.shared_state["process_status"]
                det_data = ps["detection"]
                total_pages = det_data.get("total_pages", 0)
                
                inp_data = ps["inpainting"]
                inp_data["completed_pages"] = inp_data.get("completed_pages", 0) + 1
                
                if inp_data["completed_pages"] == total_pages:
                    inp_data["status"] = "finished"
                    inp_data["current_file"] = None
                
                ps["inpainting"] = inp_data
                self.shared_state["process_status"] = ps
            
            # Unload heavy matrices securely
            page_data.unload()
            # Flaw 1 optimization: drop crops again before translation Queue (downstream only needs boxes + text).
            strip_ipc_crop_payload(page_data)
            self.out_queue.put(page_data)