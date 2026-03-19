"""maskAnnotate — Launch the napari-based mask annotation GUI."""

import os
import sys

# Auto-detect Qt plugin path for conda environments (fixes cocoa missing on macOS)
_env_plugins = os.path.join(sys.prefix, "plugins")
if os.path.isdir(_env_plugins) and "QT_PLUGIN_PATH" not in os.environ:
    os.environ["QT_PLUGIN_PATH"] = _env_plugins

import napari
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QScrollArea, QTabBar, QTabWidget

from mask_annotate.data_manager import DataManager
from mask_annotate.viewer_manager import ViewerManager
from mask_annotate.load_widget import LoadWidget
from mask_annotate.cleanup_widget import CleanupWidget
from mask_annotate.shift_widget import ShiftWidget


def main():
    viewer = napari.Viewer(title="maskAnnotate")
    data_mgr = DataManager()
    view_mgr = ViewerManager(viewer)

    # Create stage widgets, each wrapped in a scroll area so tall content
    # never overflows the dock panel regardless of screen height.
    def _scrollable(widget):
        sa = QScrollArea()
        sa.setWidget(widget)
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return sa

    load_w = LoadWidget(data_mgr)
    cleanup_w = CleanupWidget(data_mgr, view_mgr)
    shift_w = ShiftWidget(data_mgr, view_mgr)

    # Dock each widget individually, then tabify them
    dock_load = viewer.window.add_dock_widget(_scrollable(load_w), name="1. Load", area="right")
    dock_cleanup = viewer.window.add_dock_widget(_scrollable(cleanup_w), name="2. Cleanup", area="right")
    dock_shift = viewer.window.add_dock_widget(_scrollable(shift_w), name="3. Shift", area="right")

    # Tabify so they share the same panel space
    qt_win = viewer.window._qt_window
    viewer.window._qt_window.tabifyDockWidget(dock_load, dock_cleanup)
    viewer.window._qt_window.tabifyDockWidget(dock_cleanup, dock_shift)

    # Move tabs to the top so they're always visible regardless of content height
    qt_win.setTabPosition(Qt.RightDockWidgetArea, QTabWidget.North)

    dock_load.raise_()  # Show the Load tab first

    # --- Tab-switch wiring ---
    # Two independent mechanisms ensure tab changes are detected reliably:
    #   1. QDockWidget.visibilityChanged  (fires for tabified docks per Qt docs)
    #   2. QTabBar.currentChanged         (fallback, with retry until found)
    # Both funnel into _do_transition(), which guards against double-firing.

    _current_tab = [0]          # 0=Load, 1=Cleanup, 2=Shift
    _ready = [False]            # gate: ignore signals during initial window setup

    def _do_transition(new_idx):
        """Sync data when leaving a stage, activate when entering."""
        if not _ready[0] or new_idx == _current_tab[0]:
            return
        prev = _current_tab[0]

        # Sync when leaving a stage
        if prev == 0:
            load_w.apply_roi_filter()
        elif prev == 1:
            cleanup_w._sync_labels_to_data()

        # Activate when entering a stage
        if new_idx == 1:
            cleanup_w.activate()
        elif new_idx == 2:
            shift_w.activate()

        _current_tab[0] = new_idx

    # Mechanism 1: dock visibilityChanged
    dock_load.visibilityChanged.connect(lambda v: v and _do_transition(0))
    dock_cleanup.visibilityChanged.connect(lambda v: v and _do_transition(1))
    dock_shift.visibilityChanged.connect(lambda v: v and _do_transition(2))

    # Mechanism 2: QTabBar.currentChanged (retries until found)
    _tab_bar_connected = [False]

    def _on_tab_bar_changed(index, tab_bar):
        text = tab_bar.tabText(index)
        if "Load" in text:
            _do_transition(0)
        elif "Cleanup" in text:
            _do_transition(1)
        elif "Shift" in text:
            _do_transition(2)

    def _try_connect_tab_bar():
        if _tab_bar_connected[0]:
            return
        for tb in qt_win.findChildren(QTabBar):
            tab_texts = [tb.tabText(i) for i in range(tb.count())]
            if any("Load" in t for t in tab_texts):
                tb.currentChanged.connect(
                    lambda idx, _tb=tb: _on_tab_bar_changed(idx, _tb)
                )
                _tab_bar_connected[0] = True
                print(f"[maskAnnotate] Connected to QTabBar: {tab_texts}")
                return
        # Not found yet — retry
        print("[maskAnnotate] QTabBar not found, retrying in 500 ms...")
        QTimer.singleShot(500, _try_connect_tab_bar)

    # Arm the ready gate and start QTabBar search after the event loop begins
    def _on_event_loop_ready():
        _ready[0] = True
        _try_connect_tab_bar()

    QTimer.singleShot(0, _on_event_loop_ready)

    napari.run()


if __name__ == "__main__":
    main()
