"""maskAnnotate — Launch the napari-based mask annotation GUI."""

import os
import sys

# Auto-detect Qt plugin path for conda environments (fixes cocoa missing on macOS)
_env_plugins = os.path.join(sys.prefix, "plugins")
if os.path.isdir(_env_plugins) and "QT_PLUGIN_PATH" not in os.environ:
    os.environ["QT_PLUGIN_PATH"] = _env_plugins

import napari
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QTabBar

from mask_annotate.data_manager import DataManager
from mask_annotate.viewer_manager import ViewerManager
from mask_annotate.load_widget import LoadWidget
from mask_annotate.cleanup_widget import CleanupWidget
from mask_annotate.shift_widget import ShiftWidget


def main():
    viewer = napari.Viewer(title="maskAnnotate")
    data_mgr = DataManager()
    view_mgr = ViewerManager(viewer)

    # Create stage widgets
    load_w = LoadWidget(data_mgr)
    cleanup_w = CleanupWidget(data_mgr, view_mgr)
    shift_w = ShiftWidget(data_mgr, view_mgr)

    # Dock each widget individually, then tabify them
    dock_load = viewer.window.add_dock_widget(load_w, name="1. Load", area="right")
    dock_cleanup = viewer.window.add_dock_widget(cleanup_w, name="2. Cleanup", area="right")
    dock_shift = viewer.window.add_dock_widget(shift_w, name="3. Shift", area="right")

    # Tabify so they share the same panel space
    viewer.window._qt_window.tabifyDockWidget(dock_load, dock_cleanup)
    viewer.window._qt_window.tabifyDockWidget(dock_cleanup, dock_shift)
    dock_load.raise_()  # Show the Load tab first

    # Wire tab switches — sync data when leaving a stage, activate when entering.
    # Deferred with QTimer so the QTabBar is fully constructed first.
    prev_tab = [0]  # mutable so the closure can update it

    def _on_tab_changed(index, tab_bar):
        prev = prev_tab[0]
        if prev == index:
            return

        prev_text = tab_bar.tabText(prev)
        new_text = tab_bar.tabText(index)

        # Sync when leaving a stage
        if "Load" in prev_text:
            load_w.apply_roi_filter()
        elif "Cleanup" in prev_text:
            cleanup_w._sync_labels_to_data()

        # Activate when entering a stage
        if "Cleanup" in new_text:
            cleanup_w.activate()
        elif "Shift" in new_text:
            shift_w.activate()

        prev_tab[0] = index

    def _connect_tab_bar():
        qt_window = viewer.window._qt_window
        for tb in qt_window.findChildren(QTabBar):
            tab_texts = [tb.tabText(i) for i in range(tb.count())]
            if any("Load" in t for t in tab_texts):
                tb.currentChanged.connect(lambda idx, _tb=tb: _on_tab_changed(idx, _tb))
                break

    QTimer.singleShot(0, _connect_tab_bar)

    napari.run()


if __name__ == "__main__":
    main()
