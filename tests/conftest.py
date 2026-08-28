from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Settings are validated at import time; give the tests a valid baseline.
os.environ.setdefault("SERVICE_SHARED_SECRET", "test-secret-value-of-at-least-32-chars")
os.environ.setdefault(
    "DATABASE_URL", "postgresql://lawalschool:local@localhost:15432/lawalschool"
)
