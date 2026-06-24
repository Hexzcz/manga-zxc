import os
import sys
import time
import subprocess
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from multiprocessing import Queue

def get_memory_mb():
    """Gets the exact RAM usage of this python process in Megabytes on Windows."""
    pid = os.getpid()
    result = subprocess.check_output(['powershell', '-noprofile', '-command', f"(Get-Process -Id {pid}).WorkingSet64 / 1MB"])
    return float(result.decode('utf-8').strip())

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

from src.data.core import TranslationPage

def simulate_new_fastapi_upload(idx: int, filename: str, image_bytes: bytes) -> TranslationPage:
    # 1. NEW BEHAVIOR: Save raw bytes to disk (no arrays in memory!)
    test_dir = os.path.join("pages", "output", "test_spike")
    os.makedirs(test_dir, exist_ok=True)
    
    file_path = os.path.join(test_dir, f"{idx:04d}_raw.png")
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    
    # 2. Pass lightweight reference
    return TranslationPage(index=idx, file_path=file_path)


def main():
    print("--- Memory Spike Demonstration Test ---")
    input("\n[READY] Open Task Manager right now... Then press ENTER here to begin the spike! ")
    print("\nGenerating a fake 2000x3000 raw manga image (simulate user upload bytes)...")
    
    # Create a heavy image and compress to PNG bytes (Like what comes from FastAPI upload)
    fake_img = Image.new('RGB', (2000, 3000), color='white')
    img_byte_arr = BytesIO()
    fake_img.save(img_byte_arr, format='PNG')
    uploaded_bytes = img_byte_arr.getvalue()

    # The unbounded multiprocessing queue (Flaw)
    detection_queue = Queue() 

    # Clean memory explicitly before test
    import gc
    gc.collect()

    initial_memory = get_memory_mb()
    print(f"\n[INITIAL] Baseline RAM Usage: {initial_memory:.2f} MB")
    print("-" * 50)

    pages_to_upload = 50
    print(f"Simulating FastAPI 'for' loop processing {pages_to_upload} pages instantly...")

    for i in range(1, pages_to_upload + 1):
        # We now simulate the extremely lightweight pass-by-disk logic
        page_obj = simulate_new_fastapi_upload(i, f"manga_page_{i}.png", uploaded_bytes)
        detection_queue.put(page_obj)

        if i % 10 == 0:
            current_memory = get_memory_mb()
            print(f"Processed {i:2} pages | Current RAM Usage: {current_memory:.2f} MB (+{current_memory - initial_memory:.2f} MB)")
            time.sleep(0.5) # Slight delay to let Windows Task Manager graph update

    final_memory = get_memory_mb()
    print("-" * 50)
    print(f"[FINAL] Total RAM Usage: {final_memory:.2f} MB")
    print(f"[SPIKE] Memory Spiked by: {final_memory - initial_memory:.2f} MB")
    
    print("\nImagine uploading an entire volume (200 pages)! That RAM spike would quadruple and crash the server.")
    
    input("\n[PAUSED] Look at Task Manager now! The python process is holding ~3GB of RAM. Press ENTER to exit and free it...")

if __name__ == "__main__":
    main()
