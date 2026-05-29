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

from PySide6.QtWidgets import QTabWidget, QWidget

from sinner2.gui.widgets.batch_view import QBatchView
from sinner2.gui.widgets.library_view import QLibraryView
from sinner2.gui.widgets.processor_controls import QProcessorControls
from sinner2.library.media_kind import is_image, is_media
from sinner2.library.thumbnail_cache import ThumbnailCache
from sinner2.library.thumbnail_generator import ThumbnailGenerator


_SOURCES_FILTER = (
    "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All files (*)"
)
_TARGETS_FILTER = (
    "Media (*.png *.jpg *.jpeg *.mp4 *.avi *.mov *.mkv *.webm *.gif);;"
    "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;"
    "Videos (*.mp4 *.avi *.mov *.mkv *.webm);;"
    "All files (*)"
)


class QSidePanel(QTabWidget):
    """Tabbed side panel. Holds processors + libraries."""

    def __init__(
        self,
        thumbnail_cache_dir: Path,
        *,
        processors: QProcessorControls | None = None,
        batch_view: QBatchView | None = None,
        thumb_extract_dim: int = 384,
        thumb_display_dim: int = 128,
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
        self._sources_library = QLibraryView(
            self._generator,
            accept=is_image,
            file_dialog_filter=_SOURCES_FILTER,
            display_dim=thumb_display_dim,
        )
        self._targets_library = QLibraryView(
            self._generator,
            accept=is_media,
            file_dialog_filter=_TARGETS_FILTER,
            display_dim=thumb_display_dim,
        )
        self._batch_view = batch_view

        # Order: settings first (most-used during initial setup), then
        # libraries for ongoing source/target switching, then batch
        # (queue management).
        self.addTab(self._processors, "Settings")
        self.addTab(self._sources_library, "Sources")
        self.addTab(self._targets_library, "Targets")
        if self._batch_view is not None:
            self.addTab(self._batch_view, "Batch")

    # ---- Accessors ----

    def processors(self) -> QProcessorControls:
        return self._processors

    def sources_library(self) -> QLibraryView:
        return self._sources_library

    def targets_library(self) -> QLibraryView:
        return self._targets_library

    def batch_view(self) -> QBatchView | None:
        return self._batch_view

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
