from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
SOURCE_SCRIPT = THIS_DIR / "2.0.preprocessing_aftersoto.py"
DATA_ROOT = PROJECT_ROOT / "data" / "eeg"
CLEAN_ROOT = PROJECT_ROOT / "data" / "clean_EEG"
LOG_ROOT = PROJECT_ROOT / "logs" / "preprocessing_queue"

WAIT_FOR_SUBJECT = "sub48"
QUEUE_SUBJECTS = ["sub50", "sub53", "sub54"]
POLL_SECONDS = 60


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def subject_file_stems(subject: str) -> list[str]:
    vhdr_files = sorted((DATA_ROOT / subject).glob("*/*.vhdr"))
    return [path.name.replace(".vhdr", "").replace("-raw", "") for path in vhdr_files]


def expected_final_outputs(subject: str) -> list[Path]:
    return [
        CLEAN_ROOT / subject / f"reject_log_response2_ICA_{subject}_{stem}.npz"
        for stem in subject_file_stems(subject)
    ]


def missing_final_outputs(subject: str) -> list[Path]:
    return [path for path in expected_final_outputs(subject) if not path.exists()]


def wait_until_subject_finished(subject: str) -> None:
    expected = expected_final_outputs(subject)
    if not expected:
        raise RuntimeError(f"No .vhdr files found for {subject} under {DATA_ROOT / subject}")

    print(f"[{timestamp()}] Waiting for {subject} to finish in Spyder.", flush=True)
    print(f"[{timestamp()}] Expected final files: {len(expected)}", flush=True)

    while True:
        missing = missing_final_outputs(subject)
        if not missing:
            print(f"[{timestamp()}] {subject} appears complete.", flush=True)
            return

        names = ", ".join(path.name for path in missing[:4])
        print(
            f"[{timestamp()}] Still waiting for {subject}: "
            f"{len(missing)} file(s) missing: {names}",
            flush=True,
        )
        time.sleep(POLL_SECONDS)


def make_subject_script(subject: str) -> Path:
    source_text = SOURCE_SCRIPT.read_text(encoding="utf-8")
    patched_text, replacements = re.subn(
        r"subject\s*=\s*'sub\d+'",
        f"subject = '{subject}'",
        source_text,
        count=1,
    )
    if replacements != 1:
        raise RuntimeError(f"Could not patch subject line in {SOURCE_SCRIPT}")

    # Force a headless Matplotlib backend for background runs.
    marker = "import os\n"
    backend_patch = "import os\nimport matplotlib\nmatplotlib.use('Agg')\n"
    patched_text, backend_replacements = re.subn(
        re.escape(marker), backend_patch, patched_text, count=1
    )
    if backend_replacements != 1:
        raise RuntimeError(f"Could not insert Agg backend into {SOURCE_SCRIPT}")

    # Resume per session by skipping files that already have the final response2 output.
    resume_marker = (
        "        file_stem = os.path.basename(raw_file).replace('.vhdr', '').replace('-raw', '')\n"
    )
    resume_patch = (
        "        file_stem = os.path.basename(raw_file).replace('.vhdr', '').replace('-raw', '')\n"
        "        final_marker = os.path.join(\n"
        "            results_dir, f'reject_log_response2_ICA_{subject}_{file_stem}.npz'\n"
        "        )\n"
        "        if os.path.exists(final_marker):\n"
        "            print(f'[SKIP] {file_stem} already finished: {final_marker}')\n"
        "            continue\n"
    )
    patched_text, resume_replacements = re.subn(
        re.escape(resume_marker), resume_patch, patched_text, count=1
    )
    if resume_replacements != 1:
        raise RuntimeError(f"Could not insert resume logic into {SOURCE_SCRIPT}")

    queued_script = THIS_DIR / f"_queued_preprocessing_aftersoto_{subject}.py"
    queued_script.write_text(patched_text, encoding="utf-8")
    return queued_script


def run_subject(subject: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    queued_script = make_subject_script(subject)
    log_path = LOG_ROOT / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{subject}.log"

    print(f"[{timestamp()}] Starting {subject}; log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"[{timestamp()}] Running {queued_script}\n")
        log.flush()
        result = subprocess.run(
            [sys.executable, str(queued_script)],
            cwd=str(THIS_DIR),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode != 0:
        raise RuntimeError(f"{subject} failed with exit code {result.returncode}; see {log_path}")

    print(f"[{timestamp()}] Finished {subject}.", flush=True)


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    wait_until_subject_finished(WAIT_FOR_SUBJECT)

    for subject in QUEUE_SUBJECTS:
        run_subject(subject)

    print(f"[{timestamp()}] Queue complete: {', '.join(QUEUE_SUBJECTS)}", flush=True)


if __name__ == "__main__":
    main()
