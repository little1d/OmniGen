from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "dashboard" / "sample-data" / "items.json"
DEFAULT_DB_PATH = ROOT / "review-data" / "reviews.sqlite3"
FRONTEND_DIST = ROOT / "dashboard" / "dist"

REVIEW_STATUSES = ("pending", "approved", "needs_changes", "rejected")
ISSUE_OPTIONS = (
    "factual_error",
    "unsupported_claim",
    "chemistry_error",
    "geometry_error",
    "style_issue",
    "incomplete",
    "duplicate",
)


class ReviewUpdate(BaseModel):
    status: Literal["pending", "approved", "needs_changes", "rejected"]
    issues: list[str] = Field(default_factory=list)
    note: str = ""
    caption: str | None = None


class ReviewDataset:
    def __init__(self, data_path: Path, db_path: Path, dataset_id: str | None = None):
        if not data_path.is_file():
            raise FileNotFoundError(f"Review dataset not found: {data_path}")
        self.data_path = data_path
        self.dataset_id = dataset_id or self._fingerprint(data_path)
        self.items = self._load_items(data_path)
        self.by_id = {item["id"]: item for item in self.items}
        if len(self.by_id) != len(self.items):
            raise ValueError("Dataset item IDs must be unique")

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._lock = threading.Lock()
        self._initialize_db()

    @staticmethod
    def _fingerprint(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"{path.stem}-{digest.hexdigest()[:12]}"

    @staticmethod
    def _load_items(path: Path) -> list[dict[str, Any]]:
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                raw_items = [json.loads(line) for line in handle if line.strip()]
        else:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            raw_items = payload.get("items", []) if isinstance(payload, dict) else payload

        if not isinstance(raw_items, list):
            raise ValueError("Review dataset must be a JSON array, {'items': [...]}, or JSONL")

        items = []
        for index, raw in enumerate(raw_items):
            if not isinstance(raw, dict):
                raise ValueError(f"Item {index} is not a JSON object")
            item = dict(raw)
            item_id = item.get("id") or item.get("source_id")
            if item_id is None or not str(item_id).strip():
                raise ValueError(f"Item {index} must contain a stable id or source_id")
            item["id"] = str(item_id)
            item.setdefault("display_key", item["id"])
            item.setdefault("source", "unknown")
            item.setdefault("source_label", item["source"])
            item.setdefault("caption", "")
            item.setdefault("facts", {})
            item.setdefault("provenance", {})
            items.append(item)
        return items

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reviews (
                    dataset_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    issues TEXT NOT NULL DEFAULT '[]',
                    note TEXT NOT NULL DEFAULT '',
                    caption TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (dataset_id, item_id)
                )
                """
            )

    def _reviews(self) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM reviews WHERE dataset_id = ?", (self.dataset_id,)
            ).fetchall()
        return {
            row["item_id"]: {
                "status": row["status"],
                "issues": json.loads(row["issues"]),
                "note": row["note"],
                "caption": row["caption"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        }

    @staticmethod
    def _pending_review() -> dict[str, Any]:
        return {
            "status": "pending",
            "issues": [],
            "note": "",
            "caption": None,
            "updated_at": None,
        }

    def meta(self) -> dict[str, Any]:
        sources = sorted(
            {
                (str(item.get("source", "unknown")), str(item.get("source_label", "unknown")))
                for item in self.items
            }
        )
        return {
            "dataset_id": self.dataset_id,
            "data_path": str(self.data_path),
            "total": len(self.items),
            "sources": [{"value": value, "label": label} for value, label in sources],
            "statuses": REVIEW_STATUSES,
            "issue_options": ISSUE_OPTIONS,
        }

    def stats(self) -> dict[str, Any]:
        reviews = self._reviews()
        counts = {status: 0 for status in REVIEW_STATUSES}
        issue_counts = {issue: 0 for issue in ISSUE_OPTIONS}
        for item in self.items:
            review = reviews.get(item["id"], self._pending_review())
            counts[review["status"]] += 1
            for issue in review["issues"]:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        reviewed = len(self.items) - counts["pending"]
        return {
            "total": len(self.items),
            "reviewed": reviewed,
            "progress": reviewed / len(self.items) if self.items else 0,
            "counts": counts,
            "issues": issue_counts,
        }

    def list_items(
        self,
        query: str = "",
        source: str | None = None,
        status: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        reviews = self._reviews()
        needle = query.casefold().strip()
        matches = []
        for item in self.items:
            review = reviews.get(item["id"], self._pending_review())
            if source and item.get("source") != source:
                continue
            if status and review["status"] != status:
                continue
            haystack = " ".join(
                str(item.get(key, ""))
                for key in ("display_key", "source_label", "source_id", "smiles", "iupac_name", "formula")
            ).casefold()
            if needle and needle not in haystack:
                continue
            matches.append(self._summary(item, review))
        return {"items": matches[offset : offset + limit], "total": len(matches)}

    @staticmethod
    def _summary(item: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": item["id"],
            "display_key": item.get("display_key"),
            "source": item.get("source"),
            "source_label": item.get("source_label"),
            "source_id": item.get("source_id"),
            "iupac_name": item.get("iupac_name"),
            "formula": item.get("formula"),
            "smiles": item.get("smiles"),
            "caption_chars": len(
                review["caption"] if review["caption"] is not None else item.get("caption", "")
            ),
            "review": review,
        }

    def get_item(self, item_id: str) -> dict[str, Any]:
        item = self.by_id.get(item_id)
        if item is None:
            raise KeyError(item_id)
        review = self._reviews().get(item_id, self._pending_review())
        result = dict(item)
        result["review"] = review
        result["effective_caption"] = (
            review["caption"] if review["caption"] is not None else item.get("caption", "")
        )
        return result

    def save_review(self, item_id: str, update: ReviewUpdate) -> dict[str, Any]:
        if item_id not in self.by_id:
            raise KeyError(item_id)
        unknown_issues = sorted(set(update.issues) - set(ISSUE_OPTIONS))
        if unknown_issues:
            raise ValueError(f"Unknown issue types: {', '.join(unknown_issues)}")

        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO reviews (dataset_id, item_id, status, issues, note, caption, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id, item_id) DO UPDATE SET
                    status = excluded.status,
                    issues = excluded.issues,
                    note = excluded.note,
                    caption = excluded.caption,
                    updated_at = excluded.updated_at
                """,
                (
                    self.dataset_id,
                    item_id,
                    update.status,
                    json.dumps(sorted(set(update.issues)), ensure_ascii=False),
                    update.note.strip(),
                    update.caption,
                    timestamp,
                ),
            )
        return self.get_item(item_id)["review"]

    def export_jsonl(self):
        reviews = self._reviews()
        for item in self.items:
            exported = dict(item)
            review = reviews.get(item["id"], self._pending_review())
            exported["review"] = review
            if review["caption"] is not None:
                exported["caption"] = review["caption"]
            yield json.dumps(exported, ensure_ascii=False) + "\n"


def create_app(
    data_path: str | Path | None = None,
    db_path: str | Path | None = None,
    dataset_id: str | None = None,
) -> FastAPI:
    selected_data = Path(data_path or os.getenv("OMNIGEN_DATA_PATH", DEFAULT_DATA_PATH))
    selected_db = Path(db_path or os.getenv("OMNIGEN_REVIEW_DB", DEFAULT_DB_PATH))
    store = ReviewDataset(selected_data, selected_db, dataset_id or os.getenv("OMNIGEN_DATASET_ID"))

    application = FastAPI(title="OmniGen Review", version="0.1.0")
    application.state.store = store

    @application.get("/api/health")
    def health():
        return {"status": "ok", "dataset_id": store.dataset_id, "items": len(store.items)}

    @application.get("/api/meta")
    def meta():
        return store.meta()

    @application.get("/api/stats")
    def stats():
        return store.stats()

    @application.get("/api/items")
    def items(
        q: str = "",
        source: str | None = None,
        status: str | None = Query(default=None, pattern="^(pending|approved|needs_changes|rejected)$"),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=1000),
    ):
        return store.list_items(q, source, status, offset, limit)

    @application.get("/api/items/{item_id}")
    def item(item_id: str):
        try:
            return store.get_item(item_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Item not found") from error

    @application.put("/api/items/{item_id}/review")
    def review(item_id: str, update: ReviewUpdate):
        try:
            return store.save_review(item_id, update)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Item not found") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.get("/api/export")
    def export():
        headers = {"Content-Disposition": f'attachment; filename="{store.dataset_id}-reviewed.jsonl"'}
        return StreamingResponse(store.export_jsonl(), media_type="application/x-ndjson", headers=headers)

    if FRONTEND_DIST.exists():
        assets = FRONTEND_DIST / "assets"
        if assets.exists():
            application.mount("/assets", StaticFiles(directory=assets), name="assets")

        @application.get("/{path:path}", include_in_schema=False)
        def frontend(path: str):
            requested = (FRONTEND_DIST / path).resolve()
            if path.startswith("api/"):
                raise HTTPException(status_code=404, detail="API endpoint not found")
            if path and requested.is_relative_to(FRONTEND_DIST.resolve()) and requested.is_file():
                return FileResponse(requested)
            return FileResponse(FRONTEND_DIST / "index.html")
    else:
        @application.get("/", include_in_schema=False)
        def frontend_missing():
            return JSONResponse(
                {
                    "message": "Frontend is not built. Run `npm install && npm run dev` in dashboard/.",
                    "api": "/docs",
                }
            )

    return application


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run(
        "src.review_app:app",
        host=os.getenv("OMNIGEN_REVIEW_HOST", "127.0.0.1"),
        port=int(os.getenv("OMNIGEN_REVIEW_PORT", "8000")),
        reload=False,
    )
