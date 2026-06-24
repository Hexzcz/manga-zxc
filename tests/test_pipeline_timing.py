"""
Pipeline timing tests:

- Simulated workers (no ML): fast topology check.
- Real `pipeline.start_pipeline` + production workers: all images in ``pages/input/``; needs models and API key.

Run from backend: python -m unittest discover -s tests -p test_pipeline_timing.py -v
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import unittest
import uuid
from collections import defaultdict
from pathlib import Path
from multiprocessing import Process, Queue, Manager

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

import constants
from pipeline import start_pipeline
from src.data.core import TranslationPage


# --- Simulated workers (module-level for Windows spawn pickling) ---


def _sim_detection_worker(in_q: Queue, out_q: Queue, timing_log, state_lock, sleep_s: float):
    while True:
        item = in_q.get()
        if item is None:
            out_q.put(None)
            break
        t0 = time.perf_counter()
        time.sleep(sleep_s)
        dt = time.perf_counter() - t0
        with state_lock:
            timing_log.append({"stage": "detection", "page": item["index"], "dt_s": dt})
        out_q.put(item)


def _sim_recognition_worker(in_q: Queue, out_q: Queue, timing_log, state_lock, sleep_s: float):
    while True:
        item = in_q.get()
        if item is None:
            out_q.put(None)
            break
        t0 = time.perf_counter()
        time.sleep(sleep_s)
        dt = time.perf_counter() - t0
        with state_lock:
            timing_log.append({"stage": "recognition", "page": item["index"], "dt_s": dt})
        out_q.put(item)


def _sim_inpainting_worker(in_q: Queue, out_q: Queue, timing_log, state_lock, sleep_s: float):
    while True:
        item = in_q.get()
        if item is None:
            out_q.put(None)
            break
        t0 = time.perf_counter()
        time.sleep(sleep_s)
        dt = time.perf_counter() - t0
        with state_lock:
            timing_log.append({"stage": "inpainting", "page": item["index"], "dt_s": dt})
        out_q.put(item)


def _sim_translation_worker(
    in_q: Queue,
    out_q: Queue,
    shared_state,
    state_lock,
    timing_log,
    sleep_per_batch: float,
):
    """Mirrors production: buffer until total_pages, one timed batch, then release all."""
    pages: list[dict] = []
    while True:
        item = in_q.get()
        if item is None:
            break
        pages.append(item)
        with state_lock:
            total = shared_state["process_status"]["detection"]["total_pages"]
        if len(pages) == total and total > 0:
            t0 = time.perf_counter()
            time.sleep(sleep_per_batch)
            dt = time.perf_counter() - t0
            with state_lock:
                timing_log.append(
                    {"stage": "translation_batch", "pages": len(pages), "dt_s": dt}
                )
            for p in pages:
                out_q.put(p)
            pages.clear()
    out_q.put(None)


def _sim_typesetting_worker(in_q: Queue, timing_log, state_lock, sleep_s: float, done_counter):
    while True:
        item = in_q.get()
        if item is None:
            break
        t0 = time.perf_counter()
        time.sleep(sleep_s)
        dt = time.perf_counter() - t0
        with state_lock:
            timing_log.append({"stage": "typesetting", "page": item["index"], "dt_s": dt})
            done_counter.value += 1


def _aggregate_stage_times(rows: list) -> dict[str, dict[str, float]]:
    by_stage: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        st = row["stage"]
        by_stage[st].append(float(row["dt_s"]))
    out: dict[str, dict[str, float]] = {}
    for st, dts in by_stage.items():
        out[st] = {
            "count": len(dts),
            "total_s": sum(dts),
            "mean_s": sum(dts) / len(dts) if dts else 0.0,
        }
    return out


def _format_timing_report(
    agg: dict[str, dict[str, float]],
    e2e_s: float,
    num_pages: int,
    *,
    title: str = "simulated",
) -> str:
    lines = [
        "",
        f"=== Pipeline timing report ({title}) ===",
        f"Pages: {num_pages}  End-to-end (wall): {e2e_s:.4f}s",
        "Per-stage (service time):",
    ]
    order = ("detection", "recognition", "inpainting", "translation_batch", "typesetting")
    for st in order:
        if st not in agg:
            continue
        a = agg[st]
        lines.append(
            f"  {st:20s}  count={int(a['count']):3d}  "
            f"total={a['total_s']:.4f}s  mean={a['mean_s']:.4f}s"
        )
    lines.append("==========================================")
    return "\n".join(lines)


def _run_simulated_timed_pipeline(
    num_pages: int,
    sleep_detection: float,
    sleep_recognition: float,
    sleep_inpainting: float,
    sleep_translation_batch: float,
    sleep_typesetting: float,
) -> tuple[list, dict[str, dict[str, float]], float, int]:
    manager = Manager()
    timing_log = manager.list()
    state_lock = manager.Lock()
    done_counter = manager.Value("i", 0)

    shared_state = manager.dict(
        {
            "process_status": manager.dict(
                {"detection": manager.dict({"total_pages": num_pages})}
            ),
        }
    )

    q_det = Queue()
    q_rec = Queue()
    q_inp = Queue()
    q_trn = Queue()
    q_typ = Queue()

    processes = [
        Process(
            target=_sim_detection_worker,
            args=(q_det, q_rec, timing_log, state_lock, sleep_detection),
        ),
        Process(
            target=_sim_recognition_worker,
            args=(q_rec, q_inp, timing_log, state_lock, sleep_recognition),
        ),
        Process(
            target=_sim_inpainting_worker,
            args=(q_inp, q_trn, timing_log, state_lock, sleep_inpainting),
        ),
        Process(
            target=_sim_translation_worker,
            args=(
                q_trn,
                q_typ,
                shared_state,
                state_lock,
                timing_log,
                sleep_translation_batch,
            ),
        ),
        Process(
            target=_sim_typesetting_worker,
            args=(q_typ, timing_log, state_lock, sleep_typesetting, done_counter),
        ),
    ]

    for p in processes:
        p.start()

    t_wall0 = time.perf_counter()
    for i in range(num_pages):
        q_det.put({"index": i})
    q_det.put(None)

    for p in processes:
        p.join(timeout=60.0)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5.0)
            raise RuntimeError(f"Worker {p.name} did not exit cleanly")

    e2e = time.perf_counter() - t_wall0
    rows = list(timing_log)
    agg = _aggregate_stage_times(rows)
    return rows, agg, e2e, int(done_counter.value)


class TestSimulatedPipelineTiming(unittest.TestCase):
    def test_pipeline_per_stage_timers_and_summary(self):
        """Uneven stage sleeps: slow inpainting; report shows where time goes."""
        num_pages = 4
        sleep_det = 0.02
        sleep_rec = 0.02
        sleep_inp = 0.12
        sleep_trn = 0.08
        sleep_typ = 0.02

        rows, agg, e2e, finished = _run_simulated_timed_pipeline(
            num_pages,
            sleep_det,
            sleep_rec,
            sleep_inp,
            sleep_trn,
            sleep_typ,
        )
        self.assertEqual(finished, num_pages)

        self.assertEqual(
            sum(1 for r in rows if r["stage"] == "detection"),
            num_pages,
            "detection should run once per page",
        )
        self.assertEqual(
            sum(1 for r in rows if r["stage"] == "recognition"),
            num_pages,
        )
        self.assertEqual(
            sum(1 for r in rows if r["stage"] == "inpainting"),
            num_pages,
        )
        self.assertEqual(
            sum(1 for r in rows if r["stage"] == "translation_batch"),
            1,
            "translation batches all pages once",
        )
        self.assertEqual(
            sum(1 for r in rows if r["stage"] == "typesetting"),
            num_pages,
        )

        # Inpainting mean should dominate per-page stages
        self.assertGreater(
            agg["inpainting"]["mean_s"],
            agg["detection"]["mean_s"],
            "configured slow inpainting should exceed detection in mean stage time",
        )

        report = _format_timing_report(agg, e2e, num_pages, title="simulated")
        print(report)

        # Sum of recorded stage work (approx. serial CPU/sleep); wall clock can be higher on Windows
        # due to Manager/spawn overhead, so we do not assert on e2e vs that sum.
        work_total = sum(v["total_s"] for v in agg.values())
        expected_work = num_pages * (sleep_det + sleep_rec + sleep_inp) + sleep_trn + num_pages * sleep_typ
        self.assertAlmostEqual(work_total, expected_work, delta=0.15)


def _real_pipeline_resources_ok() -> bool:
    root = Path(constants.BASE_DIR)
    sb = root / constants.SPEECH_BUBBLE_MODEL_FILE_PATH
    tc = root / constants.TEXT_CLUSTER_MODEL_FILE_PATH
    mg = root / constants.MIGAN_MODEL_FILE_PATH
    key = (constants.API_KEY or "").strip()
    return sb.is_file() and tc.is_file() and mg.is_file() and bool(key)


def _prime_job_state(ctx: dict, manager, job_id: str, total: int, output_dir: str) -> None:
    """Match server `reset_shared_state` fields so translation batching and typesetting paths work."""
    with ctx["state_lock"]:
        ps = ctx["shared_state"]["process_status"]
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
        ctx["shared_state"]["process_status"] = ps
        ctx["shared_state"]["job_id"] = job_id
        ctx["shared_state"]["output_dir"] = output_dir
        ctx["shared_state"]["result_pages"] = manager.list([])


def _join_processes_deadline(processes: list, deadline_s: float) -> None:
    deadline = time.perf_counter() + deadline_s
    for p in processes:
        remaining = max(0.01, deadline - time.perf_counter())
        p.join(timeout=remaining)
    alive = [p for p in processes if p.is_alive()]
    if alive:
        for p in alive:
            p.terminate()
        for p in alive:
            p.join(timeout=10.0)
        raise RuntimeError(f"Pipeline processes did not finish: {len(alive)} still alive after {deadline_s}s")


_INPUT_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def _collect_input_folder_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    return sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _INPUT_IMAGE_EXTS
    )


@unittest.skipUnless(
    _real_pipeline_resources_ok(),
    "needs backend/model/*.pt|.onnx files and API_KEY (or provider env) in .env",
)
class TestZRealPipelineTiming(unittest.TestCase):
    """
    Imports `pipeline.start_pipeline` and runs real workers with timing_log.
    Processes every image under ``pages/input/`` (sorted by name) in one job.
    Class name uses Z prefix so unittest runs the fast simulated test class first.
    """

    def test_all_input_folder_end_to_end_timings(self):
        input_dir = Path(constants.BASE_DIR) / "pages" / "input"
        sources = _collect_input_folder_images(input_dir)
        if not sources:
            self.skipTest(
                f"No image files in {input_dir} (supported: {', '.join(sorted(_INPUT_IMAGE_EXTS))})"
            )

        n = len(sources)
        job_id = str(uuid.uuid4())
        job_dir = Path(constants.BASE_DIR) / "pages" / "output" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        raw_paths: list[Path] = []
        for idx, src in enumerate(sources):
            ext = src.suffix.lower()
            raw_path = job_dir / f"{idx:04d}_raw{ext}"
            clean_path = job_dir / f"{idx:04d}{ext}"
            shutil.copyfile(src, raw_path)
            shutil.copyfile(src, clean_path)
            raw_paths.append(raw_path)

        manager = Manager()
        timing_log = manager.list()
        ctx = start_pipeline(manager, timing_log)
        _prime_job_state(ctx, manager, job_id, total=n, output_dir=str(job_dir))

        t_wall0 = time.perf_counter()
        q = ctx["queues"]["detection"]
        for idx, raw_path in enumerate(raw_paths):
            q.put(TranslationPage(index=idx, file_path=str(raw_path)))
        q.put(None)

        # ~30–40s/page observed on a heavy page; allow headroom for API + cold models.
        deadline_s = max(900.0, float(n) * 300.0)
        _join_processes_deadline(ctx["processes"], deadline_s=deadline_s)
        e2e = time.perf_counter() - t_wall0

        rows = list(timing_log)
        agg = _aggregate_stage_times(rows)
        self.assertGreaterEqual(len(rows), 5, f"expected timing rows, got: {rows}")
        for stage in ("detection", "recognition", "inpainting", "translation_batch", "typesetting"):
            self.assertIn(stage, agg, f"missing stage {stage} in {rows}")

        self.assertEqual(agg["detection"]["count"], n, rows)
        self.assertEqual(agg["recognition"]["count"], n, rows)
        self.assertEqual(agg["inpainting"]["count"], n, rows)
        self.assertEqual(agg["translation_batch"]["count"], 1, rows)
        self.assertEqual(agg["typesetting"]["count"], n, rows)

        print(_format_timing_report(agg, e2e, n, title=f"real pipeline ({n} pages from pages/input/)"))
        print(f"Sources: {[p.name for p in sources]}")


if __name__ == "__main__":
    unittest.main()
