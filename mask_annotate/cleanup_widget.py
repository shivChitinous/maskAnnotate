import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QComboBox, QFileDialog, QGroupBox, QApplication,
)


class CleanupWidget(QWidget):
    """Stage 2: Static mask cleanup on time-aggregated reference."""

    def __init__(self, data_manager, viewer_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.viewer_manager = viewer_manager
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Reference selection ---
        ref_group = QGroupBox("Reference Image")
        ref_layout = QVBoxLayout(ref_group)

        ref_layout.addWidget(QLabel("Stack:"))
        self.combo_stack = QComboBox()
        self.combo_stack.currentTextChanged.connect(self._on_reference_changed)
        ref_layout.addWidget(self.combo_stack)

        ref_layout.addWidget(QLabel("Aggregation:"))
        self.combo_agg = QComboBox()
        self.combo_agg.addItems(["mean", "std", "max"])
        self.combo_agg.currentTextChanged.connect(self._on_reference_changed)
        ref_layout.addWidget(self.combo_agg)

        layout.addWidget(ref_group)

        # --- Status ---
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # --- Actions ---
        self.btn_clear = QPushButton("Clear all masks (redraw from scratch)")
        self.btn_clear.clicked.connect(self._on_clear_masks)
        layout.addWidget(self.btn_clear)

        self.btn_save = QPushButton("Save Cleaned Mask...")
        self.btn_save.clicked.connect(self._on_save)
        layout.addWidget(self.btn_save)

        layout.addStretch()

    def activate(self):
        """Called when this stage becomes active. Populates UI and shows layers."""
        self.combo_stack.blockSignals(True)
        self.combo_stack.clear()
        names = self.data_manager.get_stack_names()
        self.combo_stack.addItems(names)
        # Default to dffStack if available
        dff_idx = next((i for i, n in enumerate(names) if "dff" in n.lower()), 0)
        self.combo_stack.setCurrentIndex(dff_idx)
        self.combo_stack.blockSignals(False)

        # Default aggregation to max
        self.combo_agg.blockSignals(True)
        max_idx = self.combo_agg.findText("max")
        if max_idx >= 0:
            self.combo_agg.setCurrentIndex(max_idx)
        self.combo_agg.blockSignals(False)

        # Always refresh the viewer with current data
        self._on_reference_changed()

    def _on_reference_changed(self):
        stack_name = self.combo_stack.currentText()
        method = self.combo_agg.currentText()
        if not stack_name:
            return

        self.status_label.setText(f"Computing {method} of {stack_name}...")
        QApplication.processEvents()

        agg = self.data_manager.get_aggregate(stack_name, method)
        self.viewer_manager.show_image(agg, name="reference", auto_contrast=True)

        if self.data_manager.mask is not None:
            self.viewer_manager.show_labels(self.data_manager.mask, name="mask")

        self.status_label.setText("Ready — use napari label tools to edit the mask")

    def _on_clear_masks(self):
        """Zero out the mask so the user can redraw from scratch."""
        if self.data_manager.mask is not None:
            blank = np.zeros_like(self.data_manager.mask)
            self.data_manager.mask = blank
            self.viewer_manager.show_labels(blank, name="mask")
            self.status_label.setText("Masks cleared — draw new ROIs with the label tools")

    def _sync_labels_to_data(self):
        """Read the current labels layer back into data_manager.mask."""
        labels = self.viewer_manager.get_labels_data("mask")
        if labels is not None:
            self.data_manager.mask = labels

    def _on_save(self):
        labels = self.viewer_manager.get_labels_data("mask")
        if labels is None:
            self.status_label.setText("No mask to save")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Cleaned Mask", "cleaned_mask.npy",
            "NumPy files (*.npy);;All files (*)"
        )
        if path:
            self.data_manager.save_mask_3d(path, labels)
            # Update the in-memory mask to reflect edits
            self.data_manager.mask = labels
            self.status_label.setText(f"Saved to {path}")
