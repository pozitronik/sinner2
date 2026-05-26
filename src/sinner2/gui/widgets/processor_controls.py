from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sinner2.pipeline.processors.face_enhancer import FaceEnhancerParams
from sinner2.pipeline.processors.face_swapper import FaceSwapperParams
from sinner2.pipeline.skip_strategy import (
    BestEffortStrategy,
    FrameSkipStrategy,
    SyncedStrategy,
)

_STRATEGIES: dict[str, type[FrameSkipStrategy]] = {
    "Best effort (process every frame, may lag)": BestEffortStrategy,
    "Synced (skip to match wall-clock)": SyncedStrategy,
}


class QProcessorControls(QWidget):
    """Param editors for the v1 chain — FaceSwapper + FaceEnhancer.

    Emits configChanged whenever any field changes. The controller listens,
    rebuilds the chain (cheap — ONNX models are cached at module scope), and
    calls executor.set_chain() with new Processor instances.
    """

    configChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        swapper_defaults = FaceSwapperParams()
        enhancer_defaults = FaceEnhancerParams()

        swapper_box = QGroupBox("FaceSwapper")
        swapper_form = QFormLayout(swapper_box)
        self._detection_interval = QSpinBox()
        self._detection_interval.setRange(1, 30)
        self._detection_interval.setValue(swapper_defaults.detection_interval)
        self._detection_interval.valueChanged.connect(self.configChanged)
        swapper_form.addRow("Detection interval", self._detection_interval)
        self._many_faces = QCheckBox()
        self._many_faces.setChecked(swapper_defaults.many_faces)
        self._many_faces.toggled.connect(self.configChanged)
        swapper_form.addRow("Many faces", self._many_faces)

        enhancer_box = QGroupBox("FaceEnhancer (GFPGAN)")
        enhancer_box.setCheckable(True)
        enhancer_box.setChecked(True)
        enhancer_box.toggled.connect(self.configChanged)
        enhancer_form = QFormLayout(enhancer_box)
        self._upscale = QSpinBox()
        self._upscale.setRange(1, 4)
        self._upscale.setValue(enhancer_defaults.upscale)
        self._upscale.valueChanged.connect(self.configChanged)
        enhancer_form.addRow("Upscale", self._upscale)
        self._only_center_face = QCheckBox()
        self._only_center_face.setChecked(enhancer_defaults.only_center_face)
        self._only_center_face.toggled.connect(self.configChanged)
        enhancer_form.addRow("Center face only", self._only_center_face)
        self._enhancer_box = enhancer_box

        execution_box = QGroupBox("Execution")
        execution_form = QFormLayout(execution_box)
        self._strategy_combo = QComboBox()
        for label in _STRATEGIES:
            self._strategy_combo.addItem(label)
        self._strategy_combo.currentTextChanged.connect(lambda _: self.configChanged.emit())
        execution_form.addRow("Frame-skip strategy", self._strategy_combo)

        layout = QVBoxLayout(self)
        layout.addWidget(swapper_box)
        layout.addWidget(enhancer_box)
        layout.addWidget(execution_box)
        layout.addStretch()

    def swapper_params(self) -> FaceSwapperParams:
        return FaceSwapperParams(
            detection_interval=self._detection_interval.value(),
            many_faces=self._many_faces.isChecked(),
        )

    def enhancer_params(self) -> FaceEnhancerParams:
        return FaceEnhancerParams(
            upscale=self._upscale.value(),
            only_center_face=self._only_center_face.isChecked(),
        )

    def enhancer_enabled(self) -> bool:
        return self._enhancer_box.isChecked()

    def skip_strategy(self) -> FrameSkipStrategy:
        cls = _STRATEGIES[self._strategy_combo.currentText()]
        return cls()
