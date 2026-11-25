"""
Review status and comments, stored in the run directory beside the batch catalog

    <run_dir>/review_status.csv    append-only; latest row per koi wins
    <run_dir>/comments/<koi>.txt
"""

import csv
import os
from datetime import UTC, datetime

STATUSES = ("None", "Looks good", "Minor Issues", "Major Issues", "Critical Issues")
FIELDNAMES = ("time", "user", "koi_id", "status")


def _status_path(run_dir):
    return os.path.join(run_dir, "review_status.csv")


def _comments_path(run_dir, koi):
    return os.path.join(run_dir, "comments", f"{koi}.txt")


def _now():
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_statuses(run_dir):
    """{koi_id: latest status}; targets never reviewed are absent"""
    path = _status_path(run_dir)
    if not os.path.isfile(path):
        return {}
    with open(path, newline="") as f:
        return {row["koi_id"]: row["status"] for row in csv.DictReader(f)}


def append_status(run_dir, koi, status, user):
    if status not in STATUSES:
        raise ValueError(
            f"unknown review status {status!r}; expected one of {STATUSES}"
        )
    path = _status_path(run_dir)
    new = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new:
            writer.writeheader()
        writer.writerow({"time": _now(), "user": user, "koi_id": koi, "status": status})


def read_comments(run_dir, koi):
    """Comment file text; empty if there are no comments"""
    path = _comments_path(run_dir, koi)
    if not os.path.isfile(path):
        return ""
    with open(path) as f:
        return f.read()


def append_comment(run_dir, koi, text, user):
    path = _comments_path(run_dir, koi)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(f"\nDate: {_now()}\nUser: {user}\nComment: {text}\n")


def replace_comments(run_dir, koi, text):
    """Overwrite the comment file (edit feature); write then rename"""
    path = _comments_path(run_dir, koi)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)
