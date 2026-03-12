from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QSlider, QFileDialog, QGroupBox,
    QProgressBar, QApplication, QCheckBox,
)
from qtpy.QtCore import Qt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .shift_model import ShiftModel


class ShiftWidget(QWidget):
    """Stage 3: Per-timepoint, per-plane mask shifting."""

    def __init__(self, data_manager, viewer_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.viewer_manager = viewer_manager
        self.shift_model = None

        self._updating_ui = False  # guard against feedback loops
        self._dims_connected = False

        # Range-apply anchor state
        self._anchor_t = None
        self._anchor_dx = None
        self._anchor_dy = None
        self._anchor_plane = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Reference selection ---
        ref_group = QGroupBox("Reference Stack")
        ref_layout = QVBoxLayout(ref_group)
        self.combo_stack = QComboBox()
        self.combo_stack.currentTextChanged.connect(self._on_stack_changed)
        ref_layout.addWidget(self.combo_stack)
        layout.addWidget(ref_group)

        # --- Time navigation ---
        time_group = QGroupBox("Time Navigation")
        time_layout = QVBoxLayout(time_group)

        slider_row = QHBoxLayout()
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.valueChanged.connect(self._on_time_changed)
        slider_row.addWidget(self.time_slider)

        self.time_spinbox = QSpinBox()
        self.time_spinbox.setMinimum(0)
        self.time_spinbox.valueChanged.connect(self._on_time_spinbox_changed)
        slider_row.addWidget(self.time_spinbox)
        time_layout.addLayout(slider_row)

        self.time_label = QLabel("timepoint = 0")
        time_layout.addWidget(self.time_label)
        layout.addWidget(time_group)

        # --- Plane & shift controls ---
        shift_group = QGroupBox("Plane Shift (pixels)")
        shift_layout = QVBoxLayout(shift_group)

        self.plane_label = QLabel("plane: 0")
        shift_layout.addWidget(self.plane_label)

        dx_row = QHBoxLayout()
        dx_row.addWidget(QLabel("x shift:"))
        self.dx_slider = QSlider(Qt.Horizontal)
        self.dx_slider.setRange(-50, 50)
        self.dx_slider.setValue(0)
        self.dx_slider.valueChanged.connect(self._on_shift_changed)
        dx_row.addWidget(self.dx_slider)
        self.dx_label = QLabel("0")
        self.dx_label.setMinimumWidth(30)
        dx_row.addWidget(self.dx_label)
        shift_layout.addLayout(dx_row)

        dy_row = QHBoxLayout()
        dy_row.addWidget(QLabel("y shift:"))
        self.dy_slider = QSlider(Qt.Horizontal)
        self.dy_slider.setRange(-50, 50)
        self.dy_slider.setValue(0)
        self.dy_slider.valueChanged.connect(self._on_shift_changed)
        dy_row.addWidget(self.dy_slider)
        self.dy_label = QLabel("0")
        self.dy_label.setMinimumWidth(30)
        dy_row.addWidget(self.dy_label)
        shift_layout.addLayout(dy_row)

        layout.addWidget(shift_group)

        # --- Bulk actions ---
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        self.btn_copy_all_t = QPushButton("Copy shift to all timepoints (this plane)")
        self.btn_copy_all_t.clicked.connect(self._on_copy_all_timepoints)
        actions_layout.addWidget(self.btn_copy_all_t)

        self.btn_copy_all_p = QPushButton("Copy shift to all planes (this timepoint)")
        self.btn_copy_all_p.clicked.connect(self._on_copy_all_planes)
        actions_layout.addWidget(self.btn_copy_all_p)

        self.btn_copy_all_tp = QPushButton("Copy shift to all planes and timepoints")
        self.btn_copy_all_tp.clicked.connect(self._on_copy_all_planes_timepoints)
        actions_layout.addWidget(self.btn_copy_all_tp)

        self.btn_adopt_edit = QPushButton("Save plane edit to base mask")
        self.btn_adopt_edit.clicked.connect(self._on_adopt_edit)
        actions_layout.addWidget(self.btn_adopt_edit)

        self.btn_reset = QPushButton("Reset all shifts")
        self.btn_reset.clicked.connect(self._on_reset)
        actions_layout.addWidget(self.btn_reset)

        layout.addWidget(actions_group)

        # --- Range apply ---
        range_group = QGroupBox("Range Apply")
        range_layout = QVBoxLayout(range_group)

        self.chk_all_planes = QCheckBox("All planes")
        range_layout.addWidget(self.chk_all_planes)

        self.btn_anchor = QPushButton("Anchor shift here")
        self.btn_anchor.clicked.connect(self._on_anchor)
        range_layout.addWidget(self.btn_anchor)

        self.range_status = QLabel("")
        range_layout.addWidget(self.range_status)

        range_btn_row = QHBoxLayout()
        self.btn_range_cancel = QPushButton("Cancel")
        self.btn_range_cancel.setEnabled(False)
        self.btn_range_cancel.clicked.connect(self._on_range_cancel)
        range_btn_row.addWidget(self.btn_range_cancel)

        self.btn_range_apply = QPushButton("Apply range")
        self.btn_range_apply.setEnabled(False)
        self.btn_range_apply.clicked.connect(self._on_range_apply)
        range_btn_row.addWidget(self.btn_range_apply)
        range_layout.addLayout(range_btn_row)

        layout.addWidget(range_group)

        # --- Shift overview (line plot) ---
        heatmap_group = QGroupBox("Shift Overview")
        heatmap_layout = QVBoxLayout(heatmap_group)

        self._heatmap_fig = Figure(figsize=(4, 1.5), dpi=100, facecolor="black")
        self._heatmap_fig.set_tight_layout(True)
        self._heatmap_ax = self._heatmap_fig.add_subplot(111)
        self._heatmap_ax.set_facecolor("black")
        self._heatmap_canvas = FigureCanvas(self._heatmap_fig)
        self._heatmap_canvas.setFixedHeight(180)
        self._line_handles = None
        self._heatmap_vline = None

        heatmap_layout.addWidget(self._heatmap_canvas)
        layout.addWidget(heatmap_group)

        # --- Save ---
        save_group = QGroupBox("Save")
        save_layout = QVBoxLayout(save_group)

        self.btn_save = QPushButton("Save 4D Mask...")
        self.btn_save.clicked.connect(self._on_save)
        save_layout.addWidget(self.btn_save)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        save_layout.addWidget(self.progress)

        self.status_label = QLabel("")
        save_layout.addWidget(self.status_label)

        layout.addWidget(save_group)
        layout.addStretch()

    def activate(self):
        """Called when this stage becomes active."""
        self.combo_stack.blockSignals(True)
        self.combo_stack.clear()
        names = self.data_manager.get_stack_names()
        self.combo_stack.addItems(names)
        # Default to stackMC if available
        mc_idx = next((i for i, n in enumerate(names) if "mc" in n.lower()), 0)
        self.combo_stack.setCurrentIndex(mc_idx)
        self.combo_stack.blockSignals(False)

        stack_name = self.combo_stack.currentText()
        if not stack_name or self.data_manager.mask is None:
            return

        n_t = self.data_manager.get_n_timepoints(stack_name)

        # Only create a new shift model if we don't have one, or the mask/timepoints changed
        if (self.shift_model is None
                or n_t != self.shift_model.n_timepoints
                or not np.array_equal(self.shift_model.base_mask, self.data_manager.mask)):
            self.shift_model = ShiftModel(self.data_manager.mask, n_t)
            self._reset_heatmap()

        self.time_slider.setMaximum(n_t - 1)
        self.time_spinbox.setMaximum(n_t - 1)

        # Connect to napari dims event for plane tracking (only once)
        if not self._dims_connected:
            self.viewer_manager.viewer.dims.events.current_step.connect(
                self._on_plane_changed
            )
            self._dims_connected = True

        self._display_current(auto_contrast=True)
        self._update_heatmap()

    def _current_time(self):
        return self.time_slider.value()

    def _current_plane(self):
        return self.viewer_manager.get_current_plane()

    def _on_stack_changed(self):
        """Reference stack changed — reinitialize and redisplay."""
        stack_name = self.combo_stack.currentText()
        if not stack_name or self.shift_model is None:
            return
        n_t = self.data_manager.get_n_timepoints(stack_name)
        self.time_slider.setMaximum(n_t - 1)
        self.time_spinbox.setMaximum(n_t - 1)
        # Reinitialize shift model if timepoint count changed
        if n_t != self.shift_model.n_timepoints:
            self.shift_model = ShiftModel(self.data_manager.mask, n_t)
            self._reset_heatmap()
        self._display_current(auto_contrast=True)
        self._update_heatmap()

    def _on_time_changed(self, t):
        self._updating_ui = True
        self.time_spinbox.setValue(t)
        self._updating_ui = False
        self.time_label.setText(f"timepoint = {t}")
        self._display_current()
        self._update_time_marker()

    def _on_time_spinbox_changed(self, t):
        if not self._updating_ui:
            self.time_slider.setValue(t)

    def _on_plane_changed(self, event=None):
        """napari dims changed — update shift spinboxes for the new plane."""
        self._update_shift_spinboxes()
        self._update_time_marker()

    def _on_shift_changed(self):
        """User changed dx or dy spinbox — store and redisplay."""
        if self._updating_ui or self.shift_model is None:
            return
        t = self._current_time()
        p = self._current_plane()
        dx = self.dx_slider.value()
        dy = self.dy_slider.value()
        self.dx_label.setText(str(dx))
        self.dy_label.setText(str(dy))
        self.shift_model.set_shift(t, p, dx, dy)
        self._display_mask()
        self._update_heatmap()

    def _on_copy_all_timepoints(self):
        if self.shift_model is None:
            return
        p = self._current_plane()
        dx = self.dx_slider.value()
        dy = self.dy_slider.value()
        self.shift_model.set_shift_all_timepoints(p, dx, dy)
        self.status_label.setText(
            f"Copied shift ({dx}, {dy}) to all timepoints for plane {p}"
        )
        self._update_heatmap()

    def _on_copy_all_planes(self):
        if self.shift_model is None:
            return
        t = self._current_time()
        dx = self.dx_slider.value()
        dy = self.dy_slider.value()
        self.shift_model.set_shift_all_planes(t, dx, dy)
        self._display_mask()
        self.status_label.setText(
            f"Copied shift ({dx}, {dy}) to all planes for timepoint {t}"
        )
        self._update_heatmap()

    def _on_copy_all_planes_timepoints(self):
        if self.shift_model is None:
            return
        dx = self.dx_slider.value()
        dy = self.dy_slider.value()
        self.shift_model.shifts[:, :] = [dx, dy]
        self._display_mask()
        self.status_label.setText(
            f"Copied shift ({dx}, {dy}) to all planes and timepoints"
        )
        self._update_heatmap()

    def _on_adopt_edit(self):
        """Read the current plane's labels from napari, un-shift, and store as new base mask."""
        if self.shift_model is None:
            return
        t = self._current_time()
        p = self._current_plane()
        # Read what's currently displayed (user's edits included)
        edited_labels = self.viewer_manager.get_labels_data("mask")
        if edited_labels is None:
            return
        edited_plane = edited_labels[p]
        # Un-shift to get back to base mask space
        dx, dy = self.shift_model.get_shift(t, p)
        unshifted = self.shift_model._shift_plane(edited_plane, -dx, -dy)
        # Update the base mask for this plane
        self.shift_model.base_mask[p] = unshifted
        # Also update data_manager's mask so it persists if saved
        self.data_manager.mask[p] = unshifted
        self.status_label.setText(
            f"Plane {p} edit saved to base mask (will apply across all timepoints)"
        )

    def _on_reset(self):
        if self.shift_model is None:
            return
        self.shift_model.reset_all()
        self._update_shift_spinboxes()
        self._display_mask()
        self.status_label.setText("All shifts reset to (0, 0)")
        self._update_heatmap()

    def _on_anchor(self):
        """Record the current timepoint, plane, and shift as anchor."""
        if self.shift_model is None:
            return
        self._anchor_t = self._current_time()
        self._anchor_plane = self._current_plane()
        self._anchor_dx = self.dx_slider.value()
        self._anchor_dy = self.dy_slider.value()

        self.btn_anchor.setEnabled(False)
        self.btn_range_apply.setEnabled(True)
        self.btn_range_cancel.setEnabled(True)
        self.range_status.setText(
            f"Anchored at t={self._anchor_t}, plane {self._anchor_plane}, "
            f"shift ({self._anchor_dx}, {self._anchor_dy}).\n"
            f"Scroll to target and click Apply."
        )

    def _on_range_apply(self):
        """Apply the anchored shift to the range anchor_t..current_t."""
        if self.shift_model is None or self._anchor_t is None:
            return
        t_now = self._current_time()
        t_start = min(self._anchor_t, t_now)
        t_end = max(self._anchor_t, t_now)
        dx, dy = self._anchor_dx, self._anchor_dy

        if self.chk_all_planes.isChecked():
            for p in range(self.shift_model.n_planes):
                self.shift_model.set_shift_range(t_start, t_end, p, dx, dy)
            plane_msg = "all planes"
        else:
            self.shift_model.set_shift_range(
                t_start, t_end, self._anchor_plane, dx, dy
            )
            plane_msg = f"plane {self._anchor_plane}"

        self.status_label.setText(
            f"Applied shift ({dx}, {dy}) to t={t_start}..{t_end} for {plane_msg}"
        )
        self._clear_anchor()
        self._display_mask()
        self._update_shift_spinboxes()
        self._update_heatmap()

    def _on_range_cancel(self):
        """Cancel the anchored range apply."""
        self._clear_anchor()
        self.status_label.setText("Range apply cancelled")

    def _clear_anchor(self):
        """Reset anchor state and UI back to idle."""
        self._anchor_t = None
        self._anchor_dx = None
        self._anchor_dy = None
        self._anchor_plane = None
        self.btn_anchor.setEnabled(True)
        self.btn_range_apply.setEnabled(False)
        self.btn_range_cancel.setEnabled(False)
        self.range_status.setText("")

    def _display_current(self, auto_contrast=False):
        """Load and display the current timepoint's image and shifted mask."""
        stack_name = self.combo_stack.currentText()
        if not stack_name or self.shift_model is None:
            return

        t = self._current_time()
        img = self.data_manager.get_timepoint(stack_name, t)
        self.viewer_manager.show_image(img, name="reference", auto_contrast=auto_contrast)
        self._display_mask()
        self._update_shift_spinboxes()
        self._update_time_marker()

    def _display_mask(self):
        """Apply shifts and display the mask for the current timepoint.

        While an anchor is active, previews the anchored shift on the
        relevant plane(s) so the user can see where the mask will land.
        """
        if self.shift_model is None:
            return
        t = self._current_time()
        shifted = self.shift_model.apply_shifts_for_timepoint(t)

        # Preview anchored shift while in range-apply mode
        if self._anchor_t is not None:
            dx, dy = self._anchor_dx, self._anchor_dy
            if self.chk_all_planes.isChecked():
                for p in range(self.shift_model.n_planes):
                    shifted[p] = self.shift_model._shift_plane(
                        self.shift_model.base_mask[p], dx, dy
                    )
            else:
                p = self._anchor_plane
                shifted[p] = self.shift_model._shift_plane(
                    self.shift_model.base_mask[p], dx, dy
                )

        self.viewer_manager.show_labels(shifted, name="mask")

    def _update_shift_spinboxes(self):
        """Update spinbox values to reflect stored shifts for current (t, plane)."""
        if self.shift_model is None:
            return
        t = self._current_time()
        p = self._current_plane()
        self.plane_label.setText(f"plane: {p}")

        self._updating_ui = True
        dx, dy = self.shift_model.get_shift(t, p)
        self.dx_slider.setValue(dx)
        self.dy_slider.setValue(dy)
        self.dx_label.setText(str(dx))
        self.dy_label.setText(str(dy))
        self._updating_ui = False

    # ---- Heatmap helpers ------------------------------------------------

    def _reset_heatmap(self):
        """Invalidate cached line-plot objects so next update does a full redraw."""
        self._line_handles = None
        self._heatmap_vline = None
        self._heatmap_ax.clear()

    def _update_heatmap(self):
        """Recompute shift magnitudes and redraw the per-plane line plot."""
        if self.shift_model is None:
            return

        import matplotlib.cm as cm
        from matplotlib.ticker import MaxNLocator

        shifts = self.shift_model.shifts.astype(np.float32)
        # magnitudes shape: (n_timepoints, n_planes)
        magnitudes = np.sqrt(shifts[:, :, 0] ** 2 + shifts[:, :, 1] ** 2)
        n_planes = magnitudes.shape[1]
        cur_plane = self._current_plane()

        ax = self._heatmap_ax
        viridis = cm.get_cmap("viridis", n_planes)

        if self._line_handles is None:
            ax.clear()
            ax.set_facecolor("black")

            self._line_handles = []
            for p in range(n_planes):
                color = viridis(p / max(n_planes - 1, 1))
                is_cur = (p == cur_plane)
                lw = 2.0 if is_cur else 1.0
                alpha = 1.0 if is_cur else 0.5
                line, = ax.plot(
                    magnitudes[:, p],
                    color=color,
                    linewidth=lw,
                    alpha=alpha,
                )
                self._line_handles.append(line)

            ax.set_xlabel("timepoint", fontsize=8, color="white")
            ax.set_ylabel("shift (px)", fontsize=8, color="white")
            ax.tick_params(labelsize=7, colors="white")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            for spine in ax.spines.values():
                spine.set_color("white")

            t = self._current_time()
            self._heatmap_vline = ax.axvline(
                x=t, color="red", linewidth=1.0, linestyle="--"
            )
        else:
            for p, line in enumerate(self._line_handles):
                line.set_ydata(magnitudes[:, p])

        # Adjust y-axis range
        max_mag = magnitudes.max()
        ax.set_ylim(0, max(max_mag * 1.1, 1.0))

        self._heatmap_canvas.draw_idle()

    def _update_time_marker(self):
        """Move the time marker and update plane highlighting."""
        if self._heatmap_vline is None:
            return
        t = self._current_time()
        self._heatmap_vline.set_xdata([t, t])

        # Update line highlighting for current plane
        if self._line_handles is not None:
            cur_plane = self._current_plane()
            for p, line in enumerate(self._line_handles):
                if p == cur_plane:
                    line.set_linewidth(2.0)
                    line.set_alpha(1.0)
                else:
                    line.set_linewidth(1.0)
                    line.set_alpha(0.5)

        self._heatmap_canvas.draw_idle()

    # ---- Save ----------------------------------------------------------

    def _on_save(self):
        if self.shift_model is None:
            self.status_label.setText("No shift model — load data first")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save 4D Mask", "mask_4d.npy",
            "NumPy files (*.npy);;All files (*)"
        )
        if not path:
            return

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.btn_save.setEnabled(False)
        self.status_label.setText("Generating 4D mask...")
        QApplication.processEvents()

        def update_progress(frac):
            self.progress.setValue(int(frac * 100))
            QApplication.processEvents()

        shape = self.data_manager.save_mask_4d(
            path, self.shift_model, callback=update_progress
        )
        self.progress.setVisible(False)
        self.btn_save.setEnabled(True)
        self.status_label.setText(f"Saved {shape} to {path}")
