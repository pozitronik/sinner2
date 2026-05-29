"""Modal dialog for editing a BatchTask's config.

Surfaces the full session config (chain + execution + output) so the
user can tweak any aspect of how the task will run. v1 keeps the form
fields verbatim — a future refactor could share UI with QProcessorControls.

Open via .from_task(task) → user edits → accept() commits back via
.to_task() — caller persists.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sinner2.batch.task import (
    BatchCleanupMode,
    BatchOutputFormat,
    BatchTask,
    resolve_output_path,
)
from sinner2.io.video_backend import VideoBackend
from sinner2.pipeline.image_writer import ImageFormat
from sinner2.pipeline.processors.face_swapper import TargetSex


_TARGET_SEX_OPTIONS = [
    ("Both (M+F)", TargetSex.BOTH.value),
    ("Male only", TargetSex.MALE.value),
    ("Female only", TargetSex.FEMALE.value),
    ("As source face", TargetSex.AS_SOURCE.value),
]

_CLEANUP_OPTIONS = [
    ("Keep all frames", BatchCleanupMode.KEEP.value),
    ("Auto (drop consumed stages)", BatchCleanupMode.AUTO.value),
    ("Drop all when done", BatchCleanupMode.DROP_AT_END.value),
]


class QBatchTaskDialog(QDialog):
    """Edit-a-task modal."""

    def __init__(
        self,
        task: BatchTask,
        parent: QWidget | None = None,
        global_output_dir: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit batch task")
        # Default auto-size came up too narrow to read full file paths.
        self.setMinimumWidth(600)
        self._task = task
        self._global_output_dir = global_output_dir
        # The auto-derived output path currently shown in the field. Lets
        # to_task() distinguish "left on auto" (persist None) from "user
        # typed a path" (persist verbatim). Recomputed at the end of init.
        self._auto_output = ""

        # ---- Paths group ----
        paths_box = QGroupBox("Paths")
        paths_form = QFormLayout(paths_box)
        self._source_edit, source_row = self._path_picker(
            initial=str(task.source_path),
            caption="Select source",
            file_filter="Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All files (*)",
        )
        paths_form.addRow("Source:", source_row)
        self._target_edit, target_row = self._path_picker(
            initial=str(task.target_path),
            caption="Select target",
            file_filter=(
                "Media (*.png *.jpg *.jpeg *.mp4 *.avi *.mov *.mkv *.webm);;"
                "All files (*)"
            ),
        )
        paths_form.addRow("Target:", target_row)
        self._output_edit, output_row = self._path_picker(
            initial=str(task.output_path) if task.output_path else "",
            caption="Output (leave empty for default)",
            file_filter="Output file (*)",
            save_mode=True,
        )
        paths_form.addRow("Output:", output_row)
        self._format_combo = QComboBox()
        self._format_combo.addItem("Video (mp4)", BatchOutputFormat.VIDEO.value)
        self._format_combo.addItem("Frames (directory)", BatchOutputFormat.FRAMES.value)
        self._format_combo.setCurrentIndex(
            0 if task.output_format is BatchOutputFormat.VIDEO else 1
        )
        paths_form.addRow("Output format:", self._format_combo)

        # ---- FaceSwapper group ----
        swap_box = QGroupBox("FaceSwapper")
        swap_form = QFormLayout(swap_box)
        self._detection_interval = QSpinBox()
        self._detection_interval.setRange(1, 30)
        self._detection_interval.setValue(task.swapper_detection_interval)
        swap_form.addRow("Detection interval:", self._detection_interval)
        self._many_faces = QCheckBox()
        self._many_faces.setChecked(task.swapper_many_faces)
        swap_form.addRow("Many faces:", self._many_faces)
        self._target_sex = QComboBox()
        for label, value in _TARGET_SEX_OPTIONS:
            self._target_sex.addItem(label, value)
            if value == task.swapper_target_sex:
                self._target_sex.setCurrentIndex(self._target_sex.count() - 1)
        swap_form.addRow("Swap which:", self._target_sex)

        # ---- FaceEnhancer group ----
        enh_box = QGroupBox("FaceEnhancer (GFPGAN)")
        enh_box.setCheckable(True)
        enh_box.setChecked(task.enhancer_enabled)
        self._enhancer_box = enh_box
        enh_form = QFormLayout(enh_box)
        self._upscale = QSpinBox()
        self._upscale.setRange(1, 4)
        self._upscale.setValue(task.enhancer_upscale)
        enh_form.addRow("Upscale:", self._upscale)
        self._only_center_face = QCheckBox()
        self._only_center_face.setChecked(task.enhancer_only_center_face)
        enh_form.addRow("Only center face:", self._only_center_face)

        # ---- Execution group ----
        exec_box = QGroupBox("Execution")
        exec_form = QFormLayout(exec_box)
        self._worker_count = QSpinBox()
        self._worker_count.setRange(1, 16)
        self._worker_count.setValue(task.worker_count)
        exec_form.addRow("Worker count:", self._worker_count)
        self._video_backend = QComboBox()
        for backend in VideoBackend:
            self._video_backend.addItem(backend.value, backend.value)
            if backend is task.video_backend:
                self._video_backend.setCurrentIndex(
                    self._video_backend.count() - 1
                )
        exec_form.addRow("Video backend:", self._video_backend)
        self._reader_pool_size = QSpinBox()
        self._reader_pool_size.setRange(1, 16)
        self._reader_pool_size.setValue(task.reader_pool_size)
        exec_form.addRow("Reader pool size:", self._reader_pool_size)
        self._cleanup_combo = QComboBox()
        for label, value in _CLEANUP_OPTIONS:
            self._cleanup_combo.addItem(label, value)
            if value == task.cleanup_mode.value:
                self._cleanup_combo.setCurrentIndex(
                    self._cleanup_combo.count() - 1
                )
        self._cleanup_combo.setToolTip(
            "Keep: retain every stage's frames. Auto: delete a stage once "
            "the next has consumed it. Drop all when done: delete all "
            "intermediates after the final output. The output is always kept."
        )
        exec_form.addRow("Intermediate frames:", self._cleanup_combo)

        # ---- Output encoding group ----
        out_box = QGroupBox("Output encoding (frames mode + ffmpeg input)")
        out_form = QFormLayout(out_box)
        self._image_format = QComboBox()
        for fmt in ImageFormat:
            self._image_format.addItem(fmt.value, fmt.value)
            if fmt is task.image_format:
                self._image_format.setCurrentIndex(
                    self._image_format.count() - 1
                )
        out_form.addRow("Image format:", self._image_format)
        self._image_quality = QSpinBox()
        self._image_quality.setRange(1, 100)
        self._image_quality.setValue(task.image_quality)
        out_form.addRow("Image quality:", self._image_quality)

        # ---- Standard OK / Cancel ----
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(paths_box)
        layout.addWidget(swap_box)
        layout.addWidget(enh_box)
        layout.addWidget(exec_box)
        layout.addWidget(out_box)
        layout.addWidget(button_box)

        # ---- Output default ----
        # With no explicit override, show the auto-derived path (so the
        # user can see exactly where output lands) and keep it synced to
        # source/target/format edits until they type their own path.
        self._auto_output = str(self._resolve_default_output())
        if task.output_path is None:
            self._output_edit.setText(self._auto_output)
        self._output_edit.setToolTip(
            "Auto-generated from the source + target names. Edit to use a "
            "custom path; clear it to restore the automatic name."
        )
        self._source_edit.textChanged.connect(self._refresh_default_output)
        self._target_edit.textChanged.connect(self._refresh_default_output)
        self._format_combo.currentIndexChanged.connect(
            self._refresh_default_output
        )

    @classmethod
    def from_task(
        cls,
        task: BatchTask,
        parent: QWidget | None = None,
        global_output_dir: Path | None = None,
    ) -> "QBatchTaskDialog":
        return cls(task, parent=parent, global_output_dir=global_output_dir)

    def to_task(self) -> BatchTask:
        """Return a new BatchTask with edits applied. Preserves the
        original task's id + runtime state (status, last_completed_frame,
        timing). Runtime state isn't editable here."""
        # Untouched auto value (or empty) → keep output_path None so it stays
        # auto-derived (and follows source/target renames + global-output-dir
        # changes); a genuine edit is stored verbatim.
        output_str = self._output_edit.text().strip()
        output_path: Path | None
        if not output_str or output_str == self._auto_output:
            output_path = None
        else:
            output_path = Path(output_str)
        format_value = self._format_combo.currentData()
        return self._task.model_copy(
            update={
                "source_path": Path(self._source_edit.text()),
                "target_path": Path(self._target_edit.text()),
                "output_path": output_path,
                "output_format": BatchOutputFormat(format_value),
                "swapper_detection_interval": self._detection_interval.value(),
                "swapper_many_faces": self._many_faces.isChecked(),
                "swapper_target_sex": self._target_sex.currentData(),
                "enhancer_enabled": self._enhancer_box.isChecked(),
                "enhancer_upscale": self._upscale.value(),
                "enhancer_only_center_face": self._only_center_face.isChecked(),
                "worker_count": self._worker_count.value(),
                "video_backend": VideoBackend(self._video_backend.currentData()),
                "reader_pool_size": self._reader_pool_size.value(),
                "cleanup_mode": BatchCleanupMode(
                    self._cleanup_combo.currentData()
                ),
                "image_format": ImageFormat(self._image_format.currentData()),
                "image_quality": self._image_quality.value(),
            }
        )

    # ---- helpers ----

    def _resolve_default_output(self) -> Path:
        """The auto-derived output path for the current source / target /
        format, ignoring any explicit override."""
        probe = self._task.model_copy(
            update={
                "source_path": Path(self._source_edit.text()),
                "target_path": Path(self._target_edit.text()),
                "output_format": BatchOutputFormat(
                    self._format_combo.currentData()
                ),
                "output_path": None,
            }
        )
        return resolve_output_path(probe, self._global_output_dir)

    def _refresh_default_output(self) -> None:
        """Re-derive the default; if the field is still showing the old
        default (user hasn't overridden it), update it in place."""
        new_auto = str(self._resolve_default_output())
        current = self._output_edit.text().strip()
        if current == "" or current == self._auto_output:
            self._output_edit.setText(new_auto)
        self._auto_output = new_auto

    def _path_picker(
        self,
        initial: str,
        caption: str,
        file_filter: str,
        save_mode: bool = False,
    ) -> tuple[QLineEdit, QWidget]:
        """Build a (line-edit + browse-button) row. Returns the line
        edit and the composite container widget."""
        edit = QLineEdit(initial)
        btn = QPushButton("…")
        btn.setFixedWidth(28)

        def browse() -> None:
            if save_mode:
                path, _ = QFileDialog.getSaveFileName(
                    self, caption, edit.text(), file_filter
                )
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self, caption, edit.text(), file_filter
                )
            if path:
                edit.setText(path)

        btn.clicked.connect(browse)
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, stretch=1)
        layout.addWidget(btn)
        return edit, container
