from multiprocessing import Process, Queue, Manager
from multiprocessing.managers import SyncManager
from src.detection.worker import DetectionWorker
from src.recognition.worker import RecognitionWorker
from src.inpainting.worker import InpaintingWorker
from src.translation.worker import TranslationWorker
from src.typesetting.worker import TypesettingWorker
import constants


def start_pipeline(manager: SyncManager, timing_log=None):
    detection_queue = Queue()
    recognition_queue = Queue()
    inpainting_queue = Queue()
    translation_queue = Queue()
    typesetting_queue = Queue()

    shared_state = manager.dict({
        "job_id": "",
        "output_dir": "",
        "ocr_ready": False,
        "result_pages": manager.list([]),
        "process_status": manager.dict({
            "detection": manager.dict({
                "status": "waiting",
                "total_pages": None,
                "current_index": None,
                "completed_pages": 0,
                "files": manager.list([])
            }),
            "recognition": manager.dict({
                "status": "waiting",
                "current_file": None,
                "completed_pages": 0,
            }),
            "inpainting": manager.dict({
                "status": "waiting",
                "current_file": None,
                "completed_pages": 0,
            }),
            "translation": manager.dict({
                "status": "waiting",
                "current_file": None,
                "completed_pages": 0,
            }),
            "typesetting": manager.dict({
                "status": "waiting",
                "current_file": None,
                "completed_pages": 0,
            }),
        })
    })
    state_lock = manager.Lock()

    processes = [
        Process(target=detection_worker_entry, args=(
            constants.SPEECH_BUBBLE_MODEL_FILE_PATH,
            constants.TEXT_CLUSTER_MODEL_FILE_PATH,
            detection_queue,
            recognition_queue,
            shared_state,
            state_lock,
            timing_log,
        )),
        Process(target=recognition_worker_entry, args=(
            recognition_queue,
            inpainting_queue,
            shared_state,
            state_lock,
            timing_log,
        )),
        Process(target=inpainting_worker_entry, args=(
            constants.MIGAN_MODEL_FILE_PATH,
            inpainting_queue,
            translation_queue,
            shared_state,
            state_lock,
            timing_log,
        )),
        Process(target=translation_worker_entry, args=(
            constants.API_KEY, 
            constants.API_MODEL,
            translation_queue,
            typesetting_queue,
            shared_state,
            state_lock,
            timing_log,
        )),
        Process(target=typesetting_worker_entry, args=(
            typesetting_queue,
            shared_state,
            state_lock,
            timing_log,
        ))
    ]

    for p in processes:
        p.start()
    
    return {
        "manager": manager,
        "state": manager.dict(),
        "queues": {"detection": detection_queue},
        "shared_state": shared_state,
        "state_lock": state_lock,
        "processes": processes,
        "all_queues": [detection_queue, recognition_queue, inpainting_queue, translation_queue, typesetting_queue]
    }


def detection_worker_entry(
    speech_bubble_model_file_path: str,
    text_cluster_model_file_path: str,
    in_queue: Queue,
    out_queue: Queue,
    shared_state,
    state_lock,
    timing_log=None,
):
    worker = DetectionWorker(
        speech_bubble_model_file_path=speech_bubble_model_file_path,
        text_cluster_model_file_path=text_cluster_model_file_path,
        in_queue=in_queue,
        out_queue=out_queue,
        shared_state=shared_state,
        state_lock=state_lock,
        timing_log=timing_log,
    )
    worker.run()


def recognition_worker_entry(
    in_queue: Queue,
    out_queue: Queue,
    shared_state,
    state_lock,
    timing_log=None,
):
    worker = RecognitionWorker(
        in_queue=in_queue,
        out_queue=out_queue,
        shared_state=shared_state,
        state_lock=state_lock,
        timing_log=timing_log,
    )
    worker.run()


def inpainting_worker_entry(
    migan_model_file_path: str,
    in_queue: Queue,
    out_queue: Queue,
    shared_state,
    state_lock,
    timing_log=None,
):
    worker = InpaintingWorker(
        migan_model_file_path=migan_model_file_path,
        in_queue=in_queue,
        out_queue=out_queue,
        shared_state=shared_state,
        state_lock=state_lock,
        timing_log=timing_log,
    )
    worker.run()


def translation_worker_entry(
    api_key: str,
    api_model: str,
    in_queue: Queue,
    out_queue: Queue,
    shared_state,
    state_lock,
    timing_log=None,
):
    worker = TranslationWorker(
        api_key=api_key,
        api_model=api_model,
        in_queue=in_queue,
        out_queue=out_queue,
        shared_state=shared_state,
        state_lock=state_lock,
        timing_log=timing_log,
    )
    worker.run()


def typesetting_worker_entry(
    in_queue: Queue,
    shared_state,
    state_lock,
    timing_log=None,
):
    worker = TypesettingWorker(
        in_queue=in_queue,
        shared_state=shared_state,
        state_lock=state_lock,
        timing_log=timing_log,
    )
    worker.run()