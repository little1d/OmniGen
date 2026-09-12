import json

import pytest
from fastapi.testclient import TestClient

from src.review_app import create_app


def test_review_workflow(tmp_path):
    data_path = tmp_path / "items.jsonl"
    records = [
        {
            "id": "mol-1",
            "display_key": "QM9-0001",
            "source": "qm9",
            "source_label": "QM9",
            "formula": "CH4",
            "caption": "Original caption",
            "sdf": "methane\n",
        },
        {
            "id": "mol-2",
            "display_key": "QM9-0002",
            "source": "qm9",
            "source_label": "QM9",
            "formula": "H2O",
            "caption": "Water caption",
            "sdf": "water\n",
        },
    ]
    data_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    client = TestClient(create_app(data_path=data_path, db_path=tmp_path / "reviews.sqlite3"))

    assert client.get("/api/health").json()["items"] == 2
    assert client.get("/api/items", params={"status": "pending"}).json()["total"] == 2

    response = client.put(
        "/api/items/mol-1/review",
        json={
            "status": "needs_changes",
            "issues": ["factual_error"],
            "note": "Formula mismatch",
            "caption": "Corrected caption",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "needs_changes"

    detail = client.get("/api/items/mol-1").json()
    assert detail["effective_caption"] == "Corrected caption"
    assert detail["review"]["note"] == "Formula mismatch"

    stats = client.get("/api/stats").json()
    assert stats["reviewed"] == 1
    assert stats["counts"]["needs_changes"] == 1
    assert stats["issues"]["factual_error"] == 1

    exported = client.get("/api/export").text.splitlines()
    assert json.loads(exported[0])["caption"] == "Corrected caption"
    assert json.loads(exported[1])["review"]["status"] == "pending"


def test_review_rejects_unknown_issue(tmp_path):
    data_path = tmp_path / "items.json"
    data_path.write_text('[{"id": "mol-1", "caption": "Caption"}]', encoding="utf-8")
    client = TestClient(create_app(data_path=data_path, db_path=tmp_path / "reviews.sqlite3"))

    response = client.put(
        "/api/items/mol-1/review",
        json={"status": "rejected", "issues": ["not-a-real-issue"], "note": "", "caption": None},
    )
    assert response.status_code == 422


def test_review_dataset_requires_stable_ids(tmp_path):
    data_path = tmp_path / "items.json"
    data_path.write_text('[{"caption": "Caption"}]', encoding="utf-8")

    with pytest.raises(ValueError, match="stable id or source_id"):
        create_app(data_path=data_path, db_path=tmp_path / "reviews.sqlite3")


def test_dataset_fingerprint_isolates_review_namespaces(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path = first_dir / "items.json"
    second_path = second_dir / "items.json"
    first_path.write_text('[{"id": "mol-1", "caption": "First"}]', encoding="utf-8")
    second_path.write_text('[{"id": "mol-1", "caption": "Second"}]', encoding="utf-8")

    first = create_app(data_path=first_path, db_path=tmp_path / "reviews.sqlite3")
    second = create_app(data_path=second_path, db_path=tmp_path / "reviews.sqlite3")

    assert first.state.store.dataset_id != second.state.store.dataset_id
