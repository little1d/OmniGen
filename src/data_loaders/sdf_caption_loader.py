import json
from pathlib import Path
from typing import Any

from .base import BaseDataLoader
from .data_structures import DataSample, Dataset
from .registry import register_loader


@register_loader("sdf_caption")
class SDFCaptionLoader(BaseDataLoader):
    """Load paired `<molecule>.sdf` and `<molecule>.facts.json` files."""

    def can_handle(self, data_source: str | Path) -> bool:
        path = Path(data_source)
        return path.is_dir() and any(path.glob("*.sdf"))

    def load(self, data_source: str | Path, **kwargs: Any) -> Dataset:
        input_dir = Path(data_source)
        facts_dir = Path(kwargs.get("facts_dir") or input_dir)
        metadata_path = kwargs.get("metadata_path")
        if metadata_path is None:
            candidates = (input_dir / "reference.json", input_dir.parent / "reference.json")
            metadata_path = next((path for path in candidates if path.is_file()), None)
        references = self._load_references(Path(metadata_path)) if metadata_path else {}

        sdf_paths = sorted(input_dir.glob("*.sdf"))
        max_samples = kwargs.get("max_samples")
        if max_samples is not None:
            sdf_paths = sdf_paths[: int(max_samples)]
        if not sdf_paths:
            raise ValueError(f"No SDF files found in {input_dir}")

        samples = []
        for sdf_path in sdf_paths:
            facts_path = facts_dir / f"{sdf_path.stem}.facts.json"
            if not facts_path.is_file():
                raise FileNotFoundError(f"Missing verified facts: {facts_path}")
            facts = json.loads(facts_path.read_text(encoding="utf-8"))
            if not isinstance(facts, dict):
                raise ValueError(f"Verified facts must be a JSON object: {facts_path}")
            if facts.get("molecule_id") not in (None, sdf_path.stem):
                raise ValueError(f"Molecule ID does not match SDF filename: {facts_path}")
            missing = {"composition", "topology", "geometry"} - facts.keys()
            if missing:
                raise ValueError(f"Missing fact sections in {facts_path}: {sorted(missing)}")
            reference = references.get(sdf_path.stem, {})
            smiles = (
                facts.get("smiles")
                or facts.get("composition", {}).get("smiles")
                or reference.get("smiles")
            )
            if not smiles:
                raise ValueError(
                    f"Missing SMILES for {sdf_path.stem}; add it to facts or metadata_path"
                )

            samples.append(
                DataSample(
                    id=sdf_path.stem,
                    metadata={
                        "facts": facts,
                        "sdf": sdf_path.read_text(encoding="utf-8"),
                        "smiles": smiles,
                        "sdf_filename": sdf_path.name,
                        "facts_filename": facts_path.name,
                    },
                )
            )

        return Dataset(
            samples=samples,
            metadata={
                "data_type": "sdf_caption",
                "input_dir": str(input_dir),
                "facts_dir": str(facts_dir),
                "metadata_file": Path(metadata_path).name if metadata_path else None,
            },
        )

    @staticmethod
    def _load_references(path: Path) -> dict[str, dict[str, Any]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("items", []) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError(f"Metadata must be a JSON array or items object: {path}")
        references = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            molecule_id = record.get("molecule_id") or record.get("id")
            if molecule_id:
                references[str(molecule_id)] = record
        return references
