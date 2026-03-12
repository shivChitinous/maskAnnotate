from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QListWidget, QFileDialog, QGroupBox, QComboBox,
)
import numpy as np


class LoadWidget(QWidget):
    """Stage 1: File selection for 4D stacks and 3D mask."""

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self._roi_filter_applied = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- 4D Stacks ---
        stack_group = QGroupBox("4D Stacks (.nc)")
        stack_layout = QVBoxLayout(stack_group)

        self.btn_add_stack = QPushButton("Add 4D Stack...")
        self.btn_add_stack.clicked.connect(self._on_add_stack)
        stack_layout.addWidget(self.btn_add_stack)

        self.stack_list = QListWidget()
        stack_layout.addWidget(self.stack_list)
        layout.addWidget(stack_group)

        # --- 3D Mask ---
        mask_group = QGroupBox("3D Mask (.npy)")
        mask_layout = QVBoxLayout(mask_group)

        self.btn_load_mask = QPushButton("Load Mask...")
        self.btn_load_mask.clicked.connect(self._on_load_mask)
        mask_layout.addWidget(self.btn_load_mask)

        self.mask_info = QLabel("No mask loaded")
        mask_layout.addWidget(self.mask_info)
        layout.addWidget(mask_group)

        # --- ROI Filter (JSON) ---
        roi_group = QGroupBox("ROI Filter (.json) — optional")
        roi_layout = QVBoxLayout(roi_group)

        self.btn_load_json = QPushButton("Load ROI JSON...")
        self.btn_load_json.clicked.connect(self._on_load_json)
        roi_layout.addWidget(self.btn_load_json)

        roi_layout.addWidget(QLabel("ROI set to keep:"))
        self.combo_roi_set = QComboBox()
        self.combo_roi_set.addItem("(all ROIs)")
        roi_layout.addWidget(self.combo_roi_set)

        self.roi_info = QLabel("")
        roi_layout.addWidget(self.roi_info)
        layout.addWidget(roi_group)

        # --- Reload (below both mask and ROI sections) ---
        self.btn_reload_mask = QPushButton("Reload mask from disk (reset edits)")
        self.btn_reload_mask.setEnabled(False)
        self.btn_reload_mask.clicked.connect(self._on_reload_mask)
        layout.addWidget(self.btn_reload_mask)

        layout.addStretch()

    def _on_add_stack(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select 4D Stack(s)", "", "NetCDF files (*.nc);;All files (*)"
        )
        for path in paths:
            try:
                name, shape = self.data_manager.load_stack(path)
                self.stack_list.addItem(f"{name}  {shape}")
            except Exception as e:
                self.stack_list.addItem(f"ERROR: {path} — {e}")

    def _on_load_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select 3D Mask", "", "NumPy files (*.npy);;All files (*)"
        )
        if path:
            try:
                shape = self.data_manager.load_mask(path)
                self._roi_filter_applied = False
                self.mask_info.setText(f"Loaded: {path}\nShape: {shape}")
                self.btn_reload_mask.setEnabled(True)
            except Exception as e:
                self.mask_info.setText(f"Error: {e}")

    def _on_reload_mask(self):
        """Reload the mask from the original file, discarding all edits."""
        if self.data_manager.mask_path is None:
            return
        try:
            path = self.data_manager.mask_path
            shape = self.data_manager.load_mask(str(path))
            self._roi_filter_applied = False
            self.mask_info.setText(f"Reloaded: {path}\nShape: {shape}")
        except Exception as e:
            self.mask_info.setText(f"Reload error: {e}")

    def _on_load_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ROI JSON", "", "JSON files (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            roi_sets = self.data_manager.load_roi_json(path)
            self.combo_roi_set.blockSignals(True)
            self.combo_roi_set.clear()
            self.combo_roi_set.addItem("(all ROIs)")
            for name in roi_sets:
                self.combo_roi_set.addItem(f"{name} ({len(roi_sets[name])} labels)")
            self.combo_roi_set.blockSignals(False)
            self.roi_info.setText(f"Loaded: {path}")
        except Exception as e:
            self.roi_info.setText(f"Error: {e}")

    def apply_roi_filter(self):
        """Apply ROI filter to the mask (runs only once)."""
        if self._roi_filter_applied or self.data_manager.mask is None:
            return
        text = self.combo_roi_set.currentText()
        if text != "(all ROIs)" and text:
            key = text.split(" (")[0]
            keep = self.data_manager.roi_sets.get(key)
            if keep is not None:
                mask = self.data_manager.mask
                # Zero out labels not in the selected set
                mask[~np.isin(mask, keep)] = 0
                # Remap remaining labels to 1..N for cleaner napari colors
                old_labels = sorted(set(np.unique(mask)) - {0})
                remap = np.zeros(mask.max() + 1, dtype=mask.dtype)
                for new_label, old_label in enumerate(old_labels, start=1):
                    remap[old_label] = new_label
                # Use lookup table — safe against ordering issues
                self.data_manager.mask = remap[mask]
                self.mask_info.setText(
                    self.mask_info.text() + f"\nFiltered to {key}: {len(old_labels)} ROIs (relabeled 1–{len(old_labels)})"
                )
        self._roi_filter_applied = True
