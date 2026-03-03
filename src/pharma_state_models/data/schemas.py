from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetColumns:
    hcp_id: str
    time: str
    target: str

