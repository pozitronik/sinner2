"""Side-panel container: tabs for Settings + Sources/Targets libraries.

Owns:
  - One ThumbnailCache + ThumbnailGenerator shared between both library tabs
    (one disk dir, one worker pool — cheaper than per-tab instances)
  - Two QLibraryView instances (sources image-only, targets media)
  - The existing QProcessorControls instance, hosted on the Settings tab

Exposed as a single QTabWidget so main_window can drop it in where the
processors panel used to live. Public accessors for both libraries and
the processors keep main_window wiring straightforward.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QPushButton,
    QSplitter,
    QTabWidget,
    QWidget,
)

from sinner2.config.media_extensions import images_filter, media_filter
from sinner2.gui.widgets.batch_view import QBatchView
from sinner2.gui.widgets.library_view import QLibraryView
from sinner2.gui.widgets.processor_controls import QProcessorControls
from sinner2.library.media_kind import is_image, is_media
from sinner2.library.thumbnail_cache import ThumbnailCache
from sinner2.library.thumbnail_generator import ThumbnailGenerator


class QSidePanel(QTabWidget):
    """Tabbed side panel. Holds processors + libraries."""

    facesModeToggled = Signal(bool)  # face-mapping mode on/off (Sources tab)

    def __init__(
        self,
        thumbnail_cache_dir: Path,
        *,
        processors: QProcessorControls | None = None,
        batch_view: QBatchView | None = None,
        models_view: QWidget | None = None,
        live_view: QWidget | None = None,
        face_map_panel: QWidget | None = None,
        thumb_extract_dim: int = 384,
        thumb_display_dim: int = 128,
        sources_display_dim: int | None = None,
        targets_display_dim: int | None = None,
        thumb_workers: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cache = ThumbnailCache(thumbnail_cache_dir)
        # Extract once at a generous size; the view scales down at
        # paint time so the user can resize tiles live without re-
        # extracting. thumb_workers=None lets the generator auto-scale
        # to os.cpu_count() — bulk-loading a folder pegs all cores
        # instead of trickling four at a time.
        self._generator = ThumbnailGenerator(
            self._cache,
            thumb_dim=thumb_extract_dim,
            max_workers=thumb_workers,
        )
        self._processors = processors or QProcessorControls()
        # File-dialog filters are derived from the active (configurable)
        # extension sets so the picker shows the same files the libraries
        # accept on drag-drop / folder scan. Zoom is per-panel.
        self._sources_library = QLibraryView(
            self._generator,
            accept=is_image,
            file_dialog_filter=images_filter(),
            display_dim=sources_display_dim or thumb_display_dim,
        )
        self._targets_library = QLibraryView(
            self._generator,
            accept=is_media,
            file_dialog_filter=media_filter(),
            display_dim=targets_display_dim or thumb_display_dim,
        )
        self._batch_view = batch_view
        self._models_view = models_view
        self._live_view = live_view
        self._face_map_panel = face_map_panel
        self._faces_toggle: QPushButton | None = None

        # Order: settings first (most-used during initial setup), then
        # libraries for ongoing source/target switching, then batch
        # (queue management), then models (occasional management). The Faces
        # panel is NOT its own tab — it's a togglable subpanel of Sources.
        self.addTab(self._processors, "Settings")
        self._sources_tab = self._build_sources_tab()
        self.addTab(self._sources_tab, "Sources")
        self.addTab(self._targets_library, "Targets")
        if self._batch_view is not None:
            self.addTab(self._batch_view, "Batch")
        if self._models_view is not None:
            self.addTab(self._models_view, "Models")
        if self._live_view is not None:
            self.addTab(self._live_view, "Live")

    def _build_sources_tab(self) -> QWidget:
        """Sources tab = the face-map panel beside the sources library, with a
        "Face map" toggle hosted INLINE in the library's control row (next to the
        filter edit) that reveals the panel. The toggle opens the face-map EDITOR
        — its facesModeToggled signal is wired upstream to enable preview face
        picks and route source-tile clicks to the selected face."""
        if self._face_map_panel is None:
            return self._sources_library
        self._faces_toggle = QPushButton("Face map")
        self._faces_toggle.setCheckable(True)
        self._faces_toggle.setToolTip(
            "Open the face-map editor — discover people and map each to a source "
            "(file targets only). Routing playback through the map is the "
            "'Use face map' switch in the Faces settings group."
        )
        self._faces_toggle.toggled.connect(self._on_faces_toggled)
        self._sources_library.add_leading_control(self._faces_toggle)
        self._face_map_panel.setVisible(False)  # revealed by the toggle
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._face_map_panel)
        split.addWidget(self._sources_library)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        return split

    def _on_faces_toggled(self, on: bool) -> None:
        if self._face_map_panel is not None:
            self._face_map_panel.setVisible(on)
        self.facesModeToggled.emit(on)

    # ---- Accessors ----

    def processors(self) -> QProcessorControls:
        return self._processors

    def sources_library(self) -> QLibraryView:
        return self._sources_library

    def faces_mode(self) -> bool:
        """Whether face-mapping mode (the Sources-tab "Faces" toggle) is on."""
        return self._faces_toggle is not None and self._faces_toggle.isChecked()

    def open_face_map_editor(self) -> None:
        """Reveal the face-map editor: switch to the Sources tab and turn the
        toggle on (no-op when face-mapping is unavailable — e.g. live camera)."""
        if self._faces_toggle is None or not self._faces_toggle.isEnabled():
            return
        self.setCurrentWidget(self._sources_tab)
        self._faces_toggle.setChecked(True)

    def set_faces_available(self, available: bool) -> None:
        """Enable/disable the Faces toggle. Face-mapping is file-only, so the
        camera session disables it (and clears the mode if it was on)."""
        if self._faces_toggle is None:
            return
        if not available and self._faces_toggle.isChecked():
            self._faces_toggle.setChecked(False)  # emits facesModeToggled(False)
        self._faces_toggle.setEnabled(available)

    def targets_library(self) -> QLibraryView:
        return self._targets_library

    def batch_view(self) -> QBatchView | None:
        return self._batch_view

    def models_view(self) -> QWidget | None:
        return self._models_view

    def face_map_panel(self) -> QWidget | None:
        return self._face_map_panel

    def set_editing_locked(self, locked: bool) -> None:
        """Lock the editing tabs (Settings + both libraries) during a batch
        render, leaving the Batch tab interactive — DaVinci-style: you can't
        edit while rendering."""
        self._processors.setEnabled(not locked)
        self._sources_library.setEnabled(not locked)
        self._targets_library.setEnabled(not locked)
        if self._face_map_panel is not None:
            self._face_map_panel.setEnabled(not locked)

    def set_mode(self, mode: str) -> None:
        """Show only the tabs relevant to the active mode. Live hides Targets +
        Batch (no file target; batch is file-only) and reveals + selects Live;
        any other mode restores Targets/Batch and hides Live. Resolved by widget
        index so optional tabs don't shift the mapping."""
        live = mode == "live"
        ti = self.indexOf(self._targets_library)
        if ti >= 0:
            self.setTabVisible(ti, not live)
        if self._batch_view is not None:
            bi = self.indexOf(self._batch_view)
            if bi >= 0:
                self.setTabVisible(bi, not live)
        if self._live_view is not None:
            li = self.indexOf(self._live_view)
            if li >= 0:
                self.setTabVisible(li, live)
                if live:
                    self.setCurrentWidget(self._live_view)

    def set_display_dim(self, dim: int) -> None:
        """Apply the same thumbnail display size to both libraries.
        Used to keep source/target tiles in sync (resizing one should
        resize both) and to apply persisted state on startup."""
        self._sources_library.set_display_dim(dim)
        self._targets_library.set_display_dim(dim)

    def display_dim(self) -> int:
        """Effective display dim. Both libraries share state so reading
        from either is equivalent."""
        return self._sources_library.display_dim()

    # ---- Lifecycle ----

    def shutdown(self) -> None:
        """Stop background workers. Call before closing the window so
        the thumbnail thread pool + folder scans don't outlive Qt.

        Order matters: cancel folder scans FIRST so they stop submitting
        new thumbnail jobs to the pool. Then shut the pool down with
        cancel_futures so already-queued jobs don't drain on close
        (which would keep the Python process alive long after the GUI
        is gone — large folders submit thousands of jobs)."""
        self._sources_library.shutdown()
        self._targets_library.shutdown()
        self._generator.shutdown(wait=False, cancel_futures=True)
