"""Face mapping — the per-target identity catalog.

A FaceMap is the set of people (Identities) discovered in a target, plus how a
detected face is matched to one (cosine threshold) and which source each is
swapped with. Built by the analysis pass (online embedding clustering) and
edited by the user; consumed by the swapper at swap time and stored on a
BatchTask.

Embeddings are stored NORMALIZED so a match is a plain dot product. Frozen +
value-comparable so a FaceMap diffs for change detection and round-trips to JSON
(a list of identities, each carrying a 512-float centroid).

The domain is intentionally numpy-free — pure Python keeps it trivially testable
and dependency-light. The swapper builds its own (numpy) matcher from these
centroids for the per-frame hot path; the analysis pass (a background job) uses
the pure helpers here.
"""
from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum

Embedding = Sequence[float]


class UnmatchedPolicy(str, Enum):
    """What to do with a detected face that matches no catalogued identity."""

    SKIP = "skip"        # leave it un-swapped (most precise; default)
    DEFAULT = "default"  # swap it with the map's default source
    FIRST = "first"      # swap it with the first identity's source


class IdentityMode(str, Enum):
    EMBEDDING = "embedding"  # match by ArcFace cosine similarity (default)
    POSITION = "position"    # match by detection slot index (reserved; deferred)


def normalize(vec: Embedding) -> tuple[float, ...]:
    """L2-normalize an embedding to a tuple (zero vector passes through)."""
    arr = [float(x) for x in vec]
    norm = math.sqrt(sum(x * x for x in arr))
    if norm == 0.0:
        return tuple(arr)
    return tuple(x / norm for x in arr)


def cosine(a: Embedding, b: Embedding) -> float:
    """Cosine similarity of two embeddings. Assumes both normalized (a plain dot
    product); returns -1.0 for empty / mismatched-length inputs."""
    if not a or len(a) != len(b):
        return -1.0
    return sum(x * y for x, y in zip(a, b))


@dataclass(frozen=True)
class Identity:
    """One discovered person: a normalized embedding centroid + the source
    assigned to them (None until the user maps it)."""

    id: str
    centroid: tuple[float, ...]
    source_path: str | None = None
    occurrences: int = 1
    label: str | None = None

    @staticmethod
    def new(
        embedding: Embedding,
        *,
        source_path: str | None = None,
        label: str | None = None,
    ) -> "Identity":
        return Identity(
            id=uuid.uuid4().hex[:12],
            centroid=normalize(embedding),
            source_path=source_path,
            label=label,
        )

    def observed(self, embedding: Embedding) -> "Identity":
        """Fold another face of this person into the centroid (running mean,
        renormalized) and bump the occurrence count."""
        ne = normalize(embedding)
        n = self.occurrences
        blended = tuple((c * n + e) / (n + 1) for c, e in zip(self.centroid, ne))
        return replace(self, centroid=normalize(blended), occurrences=n + 1)


@dataclass(frozen=True)
class FaceMap:
    """The identity catalog for one target. Empty = no mapping (the swapper uses
    its single global source, today's behavior)."""

    identities: tuple[Identity, ...] = ()
    threshold: float = 0.5  # cosine similarity to count as the same person
    unmatched: UnmatchedPolicy = UnmatchedPolicy.SKIP
    default_source: str | None = None
    mode: IdentityMode = IdentityMode.EMBEDDING

    # ---- Construction / queries ----

    @staticmethod
    def empty() -> "FaceMap":
        return FaceMap()

    def is_empty(self) -> bool:
        return not self.identities

    def is_active(self) -> bool:
        """True when the map would actually change swap routing — at least one
        identity has a source, or a default/first policy with a default source."""
        if any(i.source_path for i in self.identities):
            return True
        return (
            self.unmatched is UnmatchedPolicy.DEFAULT
            and self.default_source is not None
        )

    def best_match(self, embedding: Embedding) -> Identity | None:
        """The identity whose centroid is nearest to ``embedding`` and at least
        ``threshold`` similar, or None. Ties go to the higher similarity."""
        ne = normalize(embedding)
        best: Identity | None = None
        best_sim = self.threshold
        for ident in self.identities:
            sim = cosine(ne, ident.centroid)
            if sim >= best_sim:
                best, best_sim = ident, sim
        return best

    def source_for(self, embedding: Embedding) -> str | None:
        """The source path to swap a detected face with, applying the unmatched
        policy. None = don't swap this face.

        - Matched an identity WITH a source → that source.
        - Matched an identity with NO source → None (you tracked this person but
          haven't assigned one — skip, don't apply the catch-all).
        - Unmatched (no identity above threshold) → the unmatched policy.
        """
        match = self.best_match(embedding)
        if match is not None:
            return match.source_path
        if self.unmatched is UnmatchedPolicy.DEFAULT:
            return self.default_source
        if self.unmatched is UnmatchedPolicy.FIRST and self.identities:
            return self.identities[0].source_path
        return None

    def assigned_sources(self) -> list[str]:
        """Distinct source paths the map references (for the swapper to prepare),
        in first-seen order, including the default."""
        seen: list[str] = []
        for ident in self.identities:
            if ident.source_path and ident.source_path not in seen:
                seen.append(ident.source_path)
        if self.default_source and self.default_source not in seen:
            seen.append(self.default_source)
        return seen

    def index_of(self, identity_id: str) -> int | None:
        for i, ident in enumerate(self.identities):
            if ident.id == identity_id:
                return i
        return None

    # ---- Edits (return a new FaceMap) ----

    def with_identity(self, identity: Identity) -> "FaceMap":
        return replace(self, identities=(*self.identities, identity))

    def without_identity(self, identity_id: str) -> "FaceMap":
        return replace(
            self,
            identities=tuple(i for i in self.identities if i.id != identity_id),
        )

    def assign_source(self, identity_id: str, source_path: str | None) -> "FaceMap":
        idents = [
            replace(i, source_path=source_path) if i.id == identity_id else i
            for i in self.identities
        ]
        return replace(self, identities=tuple(idents))

    def with_threshold(self, threshold: float) -> "FaceMap":
        return replace(self, threshold=threshold)

    def with_unmatched(
        self, policy: UnmatchedPolicy, default_source: str | None = None
    ) -> "FaceMap":
        return replace(self, unmatched=policy, default_source=default_source)

    # ---- Clustering (analysis pass) ----

    def observe(self, embedding: Embedding) -> "FaceMap":
        """Online clustering: fold ``embedding`` into the nearest identity above
        threshold, or start a new identity. Returns the updated map."""
        ne = normalize(embedding)
        best_i = -1
        best_sim = self.threshold
        for i, ident in enumerate(self.identities):
            sim = cosine(ne, ident.centroid)
            if sim >= best_sim:
                best_i, best_sim = i, sim
        if best_i < 0:
            return self.with_identity(Identity.new(embedding))
        idents = list(self.identities)
        idents[best_i] = idents[best_i].observed(embedding)
        return replace(self, identities=tuple(idents))

    # ---- Serialization (JSON-friendly dict for sidecar + BatchTask) ----

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "unmatched": self.unmatched.value,
            "default_source": self.default_source,
            "mode": self.mode.value,
            "identities": [
                {
                    "id": i.id,
                    "centroid": list(i.centroid),
                    "source_path": i.source_path,
                    "occurrences": i.occurrences,
                    "label": i.label,
                }
                for i in self.identities
            ],
        }

    @staticmethod
    def from_dict(data: dict) -> "FaceMap":
        idents = tuple(
            Identity(
                id=str(d["id"]),
                centroid=tuple(float(x) for x in d.get("centroid", [])),
                source_path=d.get("source_path"),
                occurrences=int(d.get("occurrences", 1)),
                label=d.get("label"),
            )
            for d in data.get("identities", [])
        )
        return FaceMap(
            identities=idents,
            threshold=float(data.get("threshold", 0.5)),
            unmatched=UnmatchedPolicy(data.get("unmatched", "skip")),
            default_source=data.get("default_source"),
            mode=IdentityMode(data.get("mode", "embedding")),
        )
