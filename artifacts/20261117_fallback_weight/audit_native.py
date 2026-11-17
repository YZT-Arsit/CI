"""Scoring-only JSON round-trip for native tuple versus serialized list values."""
import hashlib
import json
from pathlib import Path
import audit

R=Path(__file__).resolve().parent
plan=json.loads((R/'SCORING_CORRECTION.json').read_text())
assert all(hashlib.sha256((R/name).read_bytes()).hexdigest()==digest for name,digest in plan['sources'].items())
native_targets=audit.targets
audit.targets=lambda w:json.loads(json.dumps(native_targets(w)))
audit.main()
