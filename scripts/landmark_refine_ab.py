"""A/B validation for 2dfan4 landmark refinement on tilted faces.

Swaps the SAME frames two ways — detector keypoints vs 2dfan4-refined
keypoints — and writes side-by-side crops so the alignment-quality delta is
visible. The hypothesis: refined keypoints give a better-aligned (less
sheared / better-seated) swap on tilted or hard-pose faces.

For each requested frame index it writes, into scripts/:
    lm_ab_<idx>_detector.png   (swap with the detector's 5 kps)
    lm_ab_<idx>_refined.png    (swap with the 2dfan4-derived 5 kps)
    lm_ab_<idx>_points.png     (the frame with detector kps in red,
                                refined kps in green — see the shift)

Also prints, per face, the pixel shift between detector and refined kps and
the 2dfan4 confidence, plus the roll each angle source reports.

Run on Windows (GPU) from the project root:

    .venv\\Scripts\\python.exe scripts\\landmark_refine_ab.py <source_face> <target_video> [frame_indices_csv]

Defaults to a spread of frames; pick indices where the face is clearly tilted.
Paste the output back and eyeball the crop pairs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from sinner2.config.source import Source
from sinner2.pipeline.processors.face_swapper import FaceSwapper, FaceSwapperParams
from sinner2.pipeline.processors.face_swapper_types import RotationAngleSource
from sinner2.pipeline.processors.landmarker import roll_from_landmark_68
from sinner2.pipeline.processors.rotation_compensation import compute_roll


def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def crop_face(frame: np.ndarray, bbox, pad: float = 0.6) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - bw * pad))
    cy1 = max(0, int(y1 - bh * pad))
    cx2 = min(w, int(x2 + bw * pad))
    cy2 = min(h, int(y2 + bh * pad))
    return frame[cy1:cy2, cx1:cx2].copy()


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: landmark_refine_ab.py <source_face> <target_video> "
              "[idx_csv]")
        return 2
    source, media = Path(sys.argv[1]), Path(sys.argv[2])
    cap = cv2.VideoCapture(str(media))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if len(sys.argv) > 3:
        indices = [int(x) for x in sys.argv[3].split(",")]
    else:
        indices = [int(total * f) for f in (0.2, 0.35, 0.5, 0.65, 0.8)]
    print(f"source={source.name} target={media.name} frames={indices}")

    base = dict(rotation_compensation=False)  # isolate the alignment effect
    plain = FaceSwapper(source=Source(path=source),
                        params=FaceSwapperParams(landmark_refine=False, **base))
    plain.setup()
    refined = FaceSwapper(source=Source(path=source),
                          params=FaceSwapperParams(landmark_refine=True, **base))
    refined.setup()
    analyser = plain._analyser            # noqa: SLF001
    landmarker = refined._landmarker      # noqa: SLF001
    out_dir = Path(__file__).resolve().parent

    for idx in indices:
        frame = read_frame(cap, idx)
        if frame is None:
            print(f"  frame {idx}: unreadable")
            continue
        faces = analyser.analyse(frame)
        if not faces:
            print(f"  frame {idx}: no face")
            continue
        face = faces[0]
        det_kps = np.asarray(face.kps, np.float32).copy()
        lm68, score = landmarker.detect_68(frame, face.bbox)
        ref_kps = np.array([
            lm68[36:42].mean(0), lm68[42:48].mean(0),
            lm68[30], lm68[48], lm68[54],
        ], np.float32)
        shift = float(np.linalg.norm(det_kps - ref_kps, axis=1).mean())
        roll_kps = compute_roll(face, RotationAngleSource.KEYPOINTS)
        roll_lm = roll_from_landmark_68(lm68)
        print(f"  frame {idx}: 2dfan4 score={score:.2f}  mean kps shift="
              f"{shift:.1f}px  roll: detector={roll_kps:+.1f} 2dfan4={roll_lm:+.1f}")

        det_out = plain.process(frame.copy())
        ref_out = refined.process(frame.copy())
        cv2.imwrite(str(out_dir / f"lm_ab_{idx}_detector.png"),
                    crop_face(det_out, face.bbox))
        cv2.imwrite(str(out_dir / f"lm_ab_{idx}_refined.png"),
                    crop_face(ref_out, face.bbox))
        viz = frame.copy()
        for (x, y) in det_kps:
            cv2.circle(viz, (int(x), int(y)), 3, (0, 0, 255), -1)   # red detector
        for (x, y) in ref_kps:
            cv2.circle(viz, (int(x), int(y)), 3, (0, 255, 0), -1)   # green refined
        cv2.imwrite(str(out_dir / f"lm_ab_{idx}_points.png"),
                    crop_face(viz, face.bbox))

    cap.release()
    plain.release()
    refined.release()
    print("\nDone. Compare scripts/lm_ab_<idx>_detector.png vs _refined.png; "
          "_points.png shows the keypoint shift (red=detector, green=2dfan4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
