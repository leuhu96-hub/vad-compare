"""Read scores that were already produced elsewhere (e.g. your lib.so on the phone).

File format = your raw output: one window per line `index, start, end, score`
(also accepts `start end score`). `path_pattern` is formatted with {stem}, {id},
{category}, {Category} (capitalised) and {index} (trailing number of the stem).
Example: "results/lib/Model/{Category}_{index}.txt"
If windows overlap (end - start > hop) they are rescored per hop like the lib does.
Speed cannot be measured for this model (RTF shows as n/a).
"""
from __future__ import annotations

import os
import re

import numpy as np

from vadcmp.groundtruth import _SPLIT
from vadcmp.models.base import VADModel
from vadcmp.timeline import FrameScores, window_scores_to_hops


class PrecomputedVAD(VADModel):
    type_name = "precomputed"
    default_threshold = 0.5

    @property
    def measures_speed(self) -> bool:
        return False

    def load(self):
        if not self.cfg.get("path_pattern"):
            raise ValueError("precomputed: set path_pattern")
        self.overlap = self.cfg.get("overlap", "mean")
        self.rescore = bool(self.cfg.get("rescore_per_hop", True))

    def _file_for(self, item: dict) -> str:
        stem = os.path.splitext(os.path.basename(item["audio"]))[0]
        m = re.search(r"(\d+)$", stem)
        cat = item.get("category", "") or ""
        p = self.cfg["path_pattern"].format(
            stem=stem, id=item["id"], category=cat, Category=cat[:1].upper() + cat[1:],
            index=m.group(1) if m else "")
        p = os.path.expanduser(p)
        return p if os.path.isabs(p) else os.path.join(self.base_dir, p)

    def process(self, audio: np.ndarray) -> FrameScores:
        dur = len(audio) / 16000
        path = self._file_for(self._item)
        rows = []
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                toks = [t for t in _SPLIT.split(line.strip()) if t]
                try:
                    nums = [float(t) for t in toks]
                except ValueError:
                    continue  # header
                if len(nums) >= 3:
                    rows.append(nums[-3:])
        a = np.array(rows, dtype=np.float64).reshape(-1, 3)
        start, end, score = a[:, 0], a[:, 1], a[:, 2]
        if len(start) > 1:
            hop = float(np.median(np.diff(start)))
            win = float(np.median(end - start))
            if self.rescore and win > hop * 1.01:
                n_hops = max(1, int(np.ceil(dur / hop)))
                offset = start[0]
                hops = window_scores_to_hops(score, hop, win, n_hops, self.overlap)
                return FrameScores.from_hops(hops, hop, offset=offset, duration=dur)
        return FrameScores(start, end, score)
