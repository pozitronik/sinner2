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
and dependency-light. For the per-frame hot path the swapper builds a numpy
matcher from these centroids (``face_swapper._CatalogMatcher`` — one GEMV instead
of this per-identity loop, with a cached id index) that mirrors ``source_for``
exactly; ``best_match``/``source_for`` here remain the reference semantics, used
by the analysis pass (a background job) and tests.
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


def _bbox4(
    raw: Sequence[float] | None,
) -> tuple[float, float, float, float] | None:
    """Coerce a stored bbox list to a 4-tuple (or None)."""
    if raw is None:
        return None
    b = [float(x) for x in raw]
    return (b[0], b[1], b[2], b[3])


def cosine(a: Embedding, b: Embedding) -> float:
    """Cosine similarity of two embeddings. Assumes both normalized (a plain dot
    product); returns -1.0 for empty / mismatched-length inputs."""
    if not a or len(a) != len(b):
        return -1.0
    return sum(x * y for x, y in zip(a, b))


@dataclass(frozen=True)
class Identity:
    """One discovered person: a normalized embedding centroid + the source
    assigned to them (None until the user maps it).

    ``ref_frame`` / ``ref_bbox`` point at this person's clearest occurrence (the
    highest-scoring detection found during analysis) so the UI can extract a
    representative thumbnail from the target on demand — kept out of the catalog
    as pixels, so the persisted JSON stays small."""

    id: str
    centroid: tuple[float, ...]
    source_path: str | None = None
    occurrences: int = 1
    label: str | None = None
    ref_frame: int | None = None  # clearest occurrence — drives the thumbnail
    ref_bbox: tuple[float, float, float, float] | None = None
    first_frame: int | None = None  # EARLIEST occurrence — drives navigation
    # Demographic hints from the representative detection (only the full
    # buffalo_l pack provides them — None in fast det+rec-only mode). Display +
    # grouping only; they don't affect swap routing.
    sex: str | None = None
    age: int | None = None
    # Representative-detection metadata for the Faces table. det_score comes from
    # any detector; pitch/yaw/roll (degrees, insightface face.pose) need the full
    # pack, so they're None in fast det+rec mode. Display only — never routing.
    det_score: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None

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
    # Routing engaged because the user turned face-mapping MODE on, even before
    # any source is assigned. Transient UI state — NOT serialized (a reloaded map
    # is unarmed until the live mode re-arms it). Lets "mode on, nothing mapped"
    # show original faces instead of falling back to the single global source.
    armed: bool = False

    # ---- Construction / queries ----

    @staticmethod
    def empty() -> "FaceMap":
        return FaceMap()

    def is_empty(self) -> bool:
        return not self.identities

    def is_active(self) -> bool:
        """True when the map should route swaps per-identity (suppressing the
        single global source). Armed = the user turned face-mapping mode on, so
        routing engages even before the first assignment (every face is then
        unmapped → shows the original, not the global source). Also active when an
        identity has a source, or a default-source policy is set."""
        if self.armed:
            return True
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

    def merge(self, identity_ids: Sequence[str]) -> "FaceMap":
        """Fold the given identities into ONE (fixing over-clustering where one
        person split across several). The first listed is the survivor (keeps its
        id + position); it absorbs the others' occurrences, its centroid becomes
        the occurrence-weighted mean (renormalized), it keeps its own source (else
        the first merged-in source), the earliest first_frame, and the clearest
        (highest det_score) occurrence's reference + demographics + pose. The
        others are removed. <2 valid ids → unchanged.

        Geometry needs no rewrite: baked embeddings re-match against the new
        catalog, so a fragment's faces route to the survivor automatically."""
        ids = [i for i in identity_ids if self.index_of(i) is not None]
        if len(ids) < 2:
            return self
        keep_id = ids[0]
        members = {m.id: m for m in self.identities if m.id in set(ids)}
        keeper = members[keep_id]
        ordered = [members[i] for i in ids]
        total = sum(m.occurrences for m in ordered) or 1
        dim = len(keeper.centroid)
        summed = [0.0] * dim
        for m in ordered:
            for k in range(min(dim, len(m.centroid))):
                summed[k] += m.centroid[k] * m.occurrences
        merged_centroid = normalize(tuple(s / total for s in summed))
        source = keeper.source_path or next(
            (m.source_path for m in ordered if m.source_path), None
        )
        rep = max(ordered, key=lambda m: (m.det_score or 0.0))
        firsts = [m.first_frame for m in ordered if m.first_frame is not None]
        merged = replace(
            keeper,
            centroid=merged_centroid,
            occurrences=total,
            source_path=source,
            first_frame=min(firsts) if firsts else keeper.first_frame,
            ref_frame=rep.ref_frame, ref_bbox=rep.ref_bbox,
            sex=rep.sex, age=rep.age, det_score=rep.det_score,
            pitch=rep.pitch, yaw=rep.yaw, roll=rep.roll,
        )
        absorbed = set(ids) - {keep_id}
        return replace(
            self,
            identities=tuple(
                merged if i.id == keep_id else i
                for i in self.identities
                if i.id not in absorbed
            ),
        )

    def assign_source(self, identity_id: str, source_path: str | None) -> "FaceMap":
        idents = [
            replace(i, source_path=source_path) if i.id == identity_id else i
            for i in self.identities
        ]
        return replace(self, identities=tuple(idents))

    def with_reference(
        self,
        identity_id: str,
        ref_frame: int,
        ref_bbox: tuple[float, float, float, float],
        sex: str | None = None,
        age: int | None = None,
        first_frame: int | None = None,
        det_score: float | None = None,
        pitch: float | None = None,
        yaw: float | None = None,
        roll: float | None = None,
    ) -> "FaceMap":
        """Record an identity's representative occurrence (set by the analysis
        pass for thumbnail extraction), its first-seen frame (for navigation),
        its demographic hints, and the table metadata (detection score + pose)."""
        idents = [
            replace(
                i, ref_frame=ref_frame, ref_bbox=ref_bbox, sex=sex, age=age,
                first_frame=first_frame if first_frame is not None else i.first_frame,
                det_score=det_score, pitch=pitch, yaw=yaw, roll=roll,
            )
            if i.id == identity_id else i
            for i in self.identities
        ]
        return replace(self, identities=tuple(idents))

    def with_threshold(self, threshold: float) -> "FaceMap":
        return replace(self, threshold=threshold)

    def with_armed(self, armed: bool) -> "FaceMap":
        """Engage/disengage routing (face-mapping mode). See ``armed``/``is_active``."""
        return replace(self, armed=bool(armed))

    def with_unmatched(
        self, policy: UnmatchedPolicy, default_source: str | None = None
    ) -> "FaceMap":
        return replace(self, unmatched=policy, default_source=default_source)

    # ---- Clustering (analysis pass) ----

    def observe_with_id(self, embedding: Embedding) -> tuple["FaceMap", str]:
        """Online clustering: fold ``embedding`` into the nearest identity above
        threshold, or start a new one. Returns the updated map AND the id of the
        identity it joined (the analysis pass uses the id to track each person's
        representative occurrence without re-matching)."""
        ne = normalize(embedding)
        best_i = -1
        best_sim = self.threshold
        for i, ident in enumerate(self.identities):
            sim = cosine(ne, ident.centroid)
            if sim >= best_sim:
                best_i, best_sim = i, sim
        if best_i < 0:
            new_ident = Identity.new(embedding)
            return self.with_identity(new_ident), new_ident.id
        idents = list(self.identities)
        joined = idents[best_i].observed(embedding)
        idents[best_i] = joined
        return replace(self, identities=tuple(idents)), joined.id

    def observe(self, embedding: Embedding) -> "FaceMap":
        """Online clustering returning just the updated map (see
        ``observe_with_id``)."""
        return self.observe_with_id(embedding)[0]

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
                    "ref_frame": i.ref_frame,
                    "ref_bbox": list(i.ref_bbox) if i.ref_bbox is not None else None,
                    "first_frame": i.first_frame,
                    "sex": i.sex,
                    "age": i.age,
                    "det_score": i.det_score,
                    "pitch": i.pitch,
                    "yaw": i.yaw,
                    "roll": i.roll,
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
                ref_frame=d.get("ref_frame"),
                ref_bbox=_bbox4(d.get("ref_bbox")),
                first_frame=d.get("first_frame"),
                sex=d.get("sex"),
                age=d.get("age"),
                det_score=d.get("det_score"),
                pitch=d.get("pitch"),
                yaw=d.get("yaw"),
                roll=d.get("roll"),
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
