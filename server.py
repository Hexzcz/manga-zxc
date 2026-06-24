from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from typing import List, Optional
from pathlib import Path
import uuid
import asyncio
import json
import base64
from pydantic import BaseModel

from multiprocessing.managers import DictProxy, ListProxy

from src.data.core import TranslationPage, load_pages_from_bytes

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "pages" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FONTS_DIR = BASE_DIR / "assets" / "fonts"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve processed output images as static files
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# Serve fonts for frontend Fabric.js canvas
if FONTS_DIR.exists():
    app.mount("/fonts", StaticFiles(directory=str(FONTS_DIR)), name="fonts")

pipeline_ctx = None

def inject_pipeline(ctx):
    global pipeline_ctx
    pipeline_ctx = ctx


@app.get("/health")
async def health():
    if pipeline_ctx and "shared_state" in pipeline_ctx:
        ocr_ready = pipeline_ctx["shared_state"].get("ocr_ready", False)
        return {"status": "ok", "ocr_ready": ocr_ready}
    return {"status": "ok", "ocr_ready": False}


@app.on_event("shutdown")
def shutdown_event():
    if pipeline_ctx and "processes" in pipeline_ctx:
        print("Shutting down pipeline processes...")
        try:
            pipeline_ctx["queues"]["detection"].put(None)
        except Exception:
            pass
        for p in pipeline_ctx["processes"]:
            if p.is_alive():
                p.terminate()
                p.join(timeout=3)
        print("Pipeline processes terminated.")


def drain_queue(q):
    """Drain all pending items from a queue (safe after workers go idle)."""
    drained = 0
    while True:
        try:
            q.get_nowait()
            drained += 1
        except Exception:
            break
    if drained:
        print(f"Drained {drained} leftover items from queue.")


def reset_shared_state(job_id: str, total: int):
    """
    Reset all process_status fields before starting a new job.
    Also stores job_id and output_dir so workers know where to write.
    """
    state_lock = pipeline_ctx["state_lock"]
    manager = pipeline_ctx["manager"]
    job_output_dir = str(OUTPUT_DIR / job_id)

    with state_lock:
        ps = pipeline_ctx["shared_state"]["process_status"]

        det = ps["detection"]
        det["status"] = "waiting"
        det["total_pages"] = total
        det["current_index"] = None
        det["completed_pages"] = 0
        det["files"] = manager.list([])
        ps["detection"] = det

        for stage in ("recognition", "inpainting", "translation", "typesetting"):
            s = ps[stage]
            s["status"] = "waiting"
            s["current_file"] = None
            s["completed_pages"] = 0
            ps[stage] = s

        pipeline_ctx["shared_state"]["process_status"] = ps
        pipeline_ctx["shared_state"]["job_id"] = job_id
        pipeline_ctx["shared_state"]["output_dir"] = job_output_dir
        pipeline_ctx["shared_state"]["result_pages"] = manager.list([])


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    job_id = str(uuid.uuid4())
    total = len(files)

    # Drain leftover items from previous job's intermediate queues
    for q in pipeline_ctx["all_queues"]:
        drain_queue(q)

    # Reset state for this new job
    reset_shared_state(job_id, total)

    pipeline_ctx["state"][job_id] = {
        "status": "queued",
        "files": [f.filename for f in files],
    }

    for idx, f in enumerate(files):
        print(f"Queuing page {idx}: {f.filename}")
        content = await f.read()
        
        # Save raw bytes to disk directly instead of decoding to massive numpy arrays
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = job_output_dir / f"{idx:04d}_raw.png"
        clean_path = job_output_dir / f"{idx:04d}.png"
        
        with open(raw_path, "wb") as disk_file:
            disk_file.write(content)
            
        import shutil
        shutil.copyfile(raw_path, clean_path)
            
        # Pass a lightweight string-reference object to the multiprocessing queue 
        page = TranslationPage(index=idx, file_path=str(raw_path))
        pipeline_ctx["queues"]["detection"].put(page)

    return {"job_id": job_id}


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Return page objects with image_url and meta_url for the given job."""
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        return JSONResponse({"job_id": job_id, "pages": []})

    # Only match 4-digit numbered PNG files (e.g. 0001.png)
    page_files = sorted(job_dir.glob("[0-9][0-9][0-9][0-9].png"))
    pages = []
    for p in page_files:
        meta_file = p.with_name(f"{p.stem}_meta.json")
        raw_file = p.with_name(f"{p.stem}_raw.png")
        pages.append({
            "image_url": f"/output/{job_id}/{p.name}",
            "raw_url": f"/output/{job_id}/{raw_file.name}" if raw_file.exists() else f"/output/{job_id}/{p.name}",
            "meta_url": f"/output/{job_id}/{meta_file.name}" if meta_file.exists() else None,
        })
    return {"job_id": job_id, "pages": pages}


class SavePageRequest(BaseModel):
    image_base64: str
    metadata: Optional[dict] = None

@app.post("/result/{job_id}/{page_index}/save")
async def save_page(job_id: str, page_index: int, req: SavePageRequest):
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        return JSONResponse({"error": "Job not found"}, status_code=404)
        
    try:
        # 1. Update the flat PNG image for export
        if "," in req.image_base64:
            header, encoded = req.image_base64.split(",", 1)
        else:
            encoded = req.image_base64
            
        img_data = base64.b64decode(encoded)
        # Use the same naming convention for consistency
        output_path = job_dir / f"{page_index:04d}.png"
        
        with open(output_path, "wb") as f:
            f.write(img_data)
            
        # 2. Update the Meta JSON if provided to keep editor/reader in sync
        if req.metadata:
            meta_path = job_dir / f"{page_index:04d}_meta.json"
            # Merge logic: We update the tr_text and positions of clusters/bubbles
            # in the existing meta file using the incoming editor data.
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    actual_meta = json.load(f)
                
                # Simple overwrite for now ensures perfect sync
                # In the future we could do a granular merge
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(req.metadata, f, ensure_ascii=False, indent=2)

        return {"status": "success", "file": output_path.name}
    except Exception as e:
        print(f"Save error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


def get_serializable_state(proxy_obj):
    """Recursively convert manager proxy objects to plain Python types."""
    if isinstance(proxy_obj, (DictProxy, dict)):
        return {k: get_serializable_state(v) for k, v in proxy_obj.items()}
    elif isinstance(proxy_obj, (ListProxy, list)):
        return [get_serializable_state(item) for item in proxy_obj]
    return proxy_obj


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()

    last_state_json = None

    try:
        while True:
            raw_state = pipeline_ctx["shared_state"].get("process_status")
            current_state = get_serializable_state(raw_state)
            current_json = json.dumps(current_state, sort_keys=True)

            if current_json != last_state_json:
                await websocket.send_json({
                    "job_id": job_id,
                    "progress": {
                        "process_status": current_state
                    }
                })
                last_state_json = current_json

            # Stop polling once typesetting is finished
            if current_state.get("typesetting", {}).get("status") == "finished":
                await asyncio.sleep(0.1)
                break

            await asyncio.sleep(0.25)

    except WebSocketDisconnect:
        print(f"Client disconnected from ws/{job_id}")
    except Exception as e:
        print(f"WebSocket error for {job_id}: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        print(f"WebSocket closed for {job_id}")