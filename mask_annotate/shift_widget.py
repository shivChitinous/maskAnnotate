from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QFileDialog, QGroupBox,
    QProgressBar, QApplication, QListWidget,
)
from qtpy.QtCore import Qt, QTimer
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

        # Playback
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._step_playback)
        self._play_multiplier = 20
        self._play_direction = 1   # +1 = forward, -1 = rewind

        self._our_update = False          # True while we write to the labels layer
        self._pending_label_sync = False  # True when user has edited labels
        self._label_event_connected = False

        # Drift correction anchor state
        self._anchors: dict = {}       # plane → [(t, x, y), ...]
        self._marking_active = False
        self._points_layer = None
        self._prev_n_points = 0

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

        # Volume rate + play buttons
        rate_row = QHBoxLayout()
        rate_row.addWidget(QLabel("volume rate (Hz):"))
        self.acq_rate_spinbox = QDoubleSpinBox()
        self.acq_rate_spinbox.setRange(0.1, 1000.0)
        self.acq_rate_spinbox.setDecimals(1)
        self.acq_rate_spinbox.setValue(8.0)
        rate_row.addWidget(self.acq_rate_spinbox)
        time_layout.addLayout(rate_row)

        play_row = QHBoxLayout()
        self.btn_play_10x = QPushButton("▶ 10×")
        self.btn_play_20x = QPushButton("▶ 20×")
        self.btn_play_40x = QPushButton("▶ 40×")
        self.btn_play_10x.clicked.connect(lambda: self._on_play(10, 1))
        self.btn_play_20x.clicked.connect(lambda: self._on_play(20, 1))
        self.btn_play_40x.clicked.connect(lambda: self._on_play(40, 1))
        play_row.addWidget(self.btn_play_10x)
        play_row.addWidget(self.btn_play_20x)
        play_row.addWidget(self.btn_play_40x)
        time_layout.addLayout(play_row)

        rewind_row = QHBoxLayout()
        self.btn_rewind_10x = QPushButton("◀ 10×")
        self.btn_rewind_20x = QPushButton("◀ 20×")
        self.btn_rewind_40x = QPushButton("◀ 40×")
        self.btn_rewind_10x.clicked.connect(lambda: self._on_play(10, -1))
        self.btn_rewind_20x.clicked.connect(lambda: self._on_play(20, -1))
        self.btn_rewind_40x.clicked.connect(lambda: self._on_play(40, -1))
        rewind_row.addWidget(self.btn_rewind_10x)
        rewind_row.addWidget(self.btn_rewind_20x)
        rewind_row.addWidget(self.btn_rewind_40x)
        time_layout.addLayout(rewind_row)

        layout.addWidget(time_group)

        # --- Shift actions ---
        shift_act_group = QGroupBox("Shift Actions")
        shift_act_layout = QVBoxLayout(shift_act_group)

        self.btn_copy_plane_trace = QPushButton("Copy time trace to all planes")
        self.btn_copy_plane_trace.clicked.connect(self._on_copy_plane_trace_to_all)
        shift_act_layout.addWidget(self.btn_copy_plane_trace)

        self.btn_reset = QPushButton("Reset all shifts")
        self.btn_reset.clicked.connect(self._on_reset)
        shift_act_layout.addWidget(self.btn_reset)

        layout.addWidget(shift_act_group)

        # --- Edit actions ---
        edit_act_group = QGroupBox("Edit Actions")
        edit_act_layout = QVBoxLayout(edit_act_group)

        self.btn_adopt_edit = QPushButton("Save plane edit")
        self.btn_adopt_edit.clicked.connect(self._on_adopt_edit)
        edit_act_layout.addWidget(self.btn_adopt_edit)

        self.btn_copy_edit_all_p = QPushButton("Copy edits to all planes (this timepoint)")
        self.btn_copy_edit_all_p.clicked.connect(self._on_copy_edit_all_planes)
        edit_act_layout.addWidget(self.btn_copy_edit_all_p)

        self.btn_reset_to_base = QPushButton("Reset to base mask")
        self.btn_reset_to_base.clicked.connect(self._on_reset_to_base)
        edit_act_layout.addWidget(self.btn_reset_to_base)

        layout.addWidget(edit_act_group)

        # --- Other actions ---
        other_act_group = QGroupBox("Other Actions")
        other_act_layout = QVBoxLayout(other_act_group)

        self.btn_push_to_base = QPushButton("Push to base mask (all planes)")
        self.btn_push_to_base.clicked.connect(self._on_push_to_base)
        other_act_layout.addWidget(self.btn_push_to_base)

        self.btn_push_plane_to_base = QPushButton("Push to base mask (this plane)")
        self.btn_push_plane_to_base.clicked.connect(self._on_push_plane_to_base)
        other_act_layout.addWidget(self.btn_push_plane_to_base)

        layout.addWidget(other_act_group)

        # --- Drift correction ---
        drift_group = QGroupBox("Drift Correction")
        drift_layout = QVBoxLayout(drift_group)

        drift_btn_row = QHBoxLayout()
        self.btn_start_marking = QPushButton("Start marking")
        self.btn_start_marking.clicked.connect(self._on_start_marking)
        drift_btn_row.addWidget(self.btn_start_marking)
        self.btn_clear_plane_anchors = QPushButton("Clear plane anchors")
        self.btn_clear_plane_anchors.clicked.connect(self._on_clear_plane_anchors)
        drift_btn_row.addWidget(self.btn_clear_plane_anchors)
        drift_layout.addLayout(drift_btn_row)

        tau_row = QHBoxLayout()
        tau_row.addWidget(QLabel("filter τ (s):"))
        self.tau_spinbox = QDoubleSpinBox()
        self.tau_spinbox.setRange(0.0, 120.0)
        self.tau_spinbox.setDecimals(2)
        self.tau_spinbox.setSingleStep(0.1)
        self.tau_spinbox.setValue(0.5)
        self.tau_spinbox.setToolTip(
            "Time constant for backwards exponential smoothing.\n"
            "0 = no filtering.  Increase to propagate each anchor\n"
            "further back in time (compensates for click delay)."
        )
        self.tau_spinbox.valueChanged.connect(self._on_tau_changed)
        tau_row.addWidget(self.tau_spinbox)
        drift_layout.addLayout(tau_row)

        self.anchor_status = QLabel("")
        self.anchor_status.setWordWrap(True)
        drift_layout.addWidget(self.anchor_status)

        self.anchor_list = QListWidget()
        self.anchor_list.setMaximumHeight(100)
        drift_layout.addWidget(self.anchor_list)

        self.btn_remove_anchor = QPushButton("Remove selected anchor")
        self.btn_remove_anchor.clicked.connect(self._on_remove_anchor)
        drift_layout.addWidget(self.btn_remove_anchor)

        layout.addWidget(drift_group)

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
        self._stop_playback()
        self._stop_marking()
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
            # Underlying data changed — discard anchors and points layer
            self._anchors.clear()
            self._refresh_anchor_list()
            try:
                self.viewer_manager.viewer.layers.remove("_drift_anchors")
            except Exception:
                pass
            self._points_layer = None

        # Re-hook label events in case the labels layer was recreated
        self._label_event_connected = False
        self._pending_label_sync = False

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
        """napari dims changed — update time marker and anchor list for the new plane."""
        self._update_time_marker()
        self._refresh_anchor_list()

    def _on_copy_plane_trace_to_all(self):
        """Copy the current plane's full time-trace of shifts to every other plane."""
        if self.shift_model is None:
            return
        p = self._current_plane()
        trace = self.shift_model.shifts[:, p, :].copy()   # (n_timepoints, 2)
        self.shift_model.shifts[:, :, :] = trace[:, np.newaxis, :]
        self._display_mask()
        self.status_label.setText(
            f"Shift time trace from plane {p} copied to all planes"
        )
        self._update_heatmap()

    def _on_adopt_edit(self):
        """Sync current napari edits into the per-timepoint mask store."""
        if self.shift_model is None:
            return
        self._pending_label_sync = True   # force sync even if flag wasn't set
        self._sync_user_edits()
        t = self._current_time()
        p = self._current_plane()
        self.status_label.setText(
            f"Edits at timepoint {t} saved (plane {p} and all other planes)"
        )

    def _on_copy_edit_all_planes(self):
        """Copy the current plane's edited mask to all planes at this timepoint."""
        if self.shift_model is None:
            return
        self._pending_label_sync = True
        self._sync_user_edits()
        t = self._current_time()
        p = self._current_plane()
        src = self.shift_model._plane_masks.get(
            (t, p), self.shift_model.base_mask[p]
        )
        for p2 in range(self.shift_model.n_planes):
            self.shift_model.set_plane_mask(t, p2, src)
        self._display_mask()
        self.status_label.setText(
            f"plane {p} edits copied to all planes at timepoint {t}"
        )

    def _on_push_to_base(self):
        """Write all edited planes at this timepoint into base_mask."""
        if self.shift_model is None:
            return
        self._pending_label_sync = True
        self._sync_user_edits()
        t = self._current_time()
        for p in range(self.shift_model.n_planes):
            src = self.shift_model._plane_masks.get(
                (t, p), self.shift_model.base_mask[p]
            )
            self.shift_model.base_mask[p] = src.copy()
        self.data_manager.mask[:] = self.shift_model.base_mask
        self.status_label.setText(
            f"all planes at timepoint {t} pushed to base mask"
        )

    def _on_push_plane_to_base(self):
        """Write only the current plane's edit at this timepoint into base_mask."""
        if self.shift_model is None:
            return
        self._pending_label_sync = True
        self._sync_user_edits()
        t = self._current_time()
        p = self._current_plane()
        src = self.shift_model._plane_masks.get(
            (t, p), self.shift_model.base_mask[p]
        )
        self.shift_model.base_mask[p] = src.copy()
        self.data_manager.mask[p] = self.shift_model.base_mask[p]
        self.status_label.setText(
            f"plane {p} at timepoint {t} pushed to base mask"
        )

    def _on_reset_to_base(self):
        """Remove all per-plane overrides for this timepoint, reverting to base_mask."""
        if self.shift_model is None:
            return
        t = self._current_time()
        self.shift_model.clear_timepoint(t)
        self._pending_label_sync = False
        self._display_mask()
        self.status_label.setText(f"timepoint {t} reset to base mask")

    def _on_reset(self):
        if self.shift_model is None:
            return
        self.shift_model.reset_all()
        self._anchors.clear()
        self._refresh_anchor_list()
        self.anchor_status.setText("")
        self._stop_marking()
        try:
            self.viewer_manager.viewer.layers.remove("_drift_anchors")
        except Exception:
            pass
        self._points_layer = None
        self._display_mask()
        self.status_label.setText("All shifts reset to (0, 0)")
        self._update_heatmap()

    # ---- Playback ----------------------------------------------------------

    def _on_play(self, multiplier, direction):
        """Start/switch/pause playback.

        direction: +1 = forward, -1 = rewind.
        Clicking the active button (same multiplier + direction) pauses.
        """
        acq_hz = self.acq_rate_spinbox.value()
        interval_ms = max(1, round(1000 / (multiplier * acq_hz)))
        if (self._play_timer.isActive()
                and self._play_multiplier == multiplier
                and self._play_direction == direction):
            self._stop_playback()
        else:
            self._play_multiplier = multiplier
            self._play_direction = direction
            self._play_timer.setInterval(interval_ms)
            self._play_timer.start()
            self._update_play_buttons()

    def _step_playback(self):
        """Advance or rewind one timepoint; stop automatically at the boundary."""
        t = self._current_time()
        if self._play_direction > 0:
            if t >= self.time_slider.maximum():
                self._stop_playback()
                return
            self.time_slider.setValue(t + 1)
        else:
            if t <= 0:
                self._stop_playback()
                return
            self.time_slider.setValue(t - 1)

    def _stop_playback(self):
        """Stop playback and reset button labels."""
        self._play_timer.stop()
        self._update_play_buttons()

    def _update_play_buttons(self):
        """Reflect current playing/paused state in button text."""
        active = self._play_timer.isActive()
        m = self._play_multiplier
        d = self._play_direction
        self.btn_play_10x.setText("⏸ 10×" if (active and m == 10 and d == 1)  else "▶ 10×")
        self.btn_play_20x.setText("⏸ 20×" if (active and m == 20 and d == 1)  else "▶ 20×")
        self.btn_play_40x.setText("⏸ 40×" if (active and m == 40 and d == 1)  else "▶ 40×")
        self.btn_rewind_10x.setText("⏸ 10×" if (active and m == 10 and d == -1) else "◀ 10×")
        self.btn_rewind_20x.setText("⏸ 20×" if (active and m == 20 and d == -1) else "◀ 20×")
        self.btn_rewind_40x.setText("⏸ 40×" if (active and m == 40 and d == -1) else "◀ 40×")

    # ---- Drift correction --------------------------------------------------

    def _on_start_marking(self):
        """Toggle drift-correction marking mode on/off."""
        if self._marking_active:
            self._stop_marking()
            return
        self._marking_active = True
        self.btn_start_marking.setText("Stop marking")
        self.anchor_status.setText(
            "Select the Points layer, then click a landmark in the viewer. "
            "First click per plane = reference (shift 0)."
        )
        # Create (or reuse) a Points layer for capturing positions
        viewer = self.viewer_manager.viewer
        try:
            self._points_layer = viewer.layers["_drift_anchors"]
        except KeyError:
            # Use an explicit empty array shaped to the viewer's dimensionality
            # so napari never has to insert NaN for missing dimensions.
            n_dims = viewer.dims.ndim
            empty = np.empty((0, n_dims), dtype=float)
            _pts_kwargs = dict(name="_drift_anchors", size=10, face_color="yellow")
            try:
                self._points_layer = viewer.add_points(
                    empty, border_color="white", **_pts_kwargs
                )
            except TypeError:
                self._points_layer = viewer.add_points(
                    empty, edge_color="white", **_pts_kwargs
                )
        self._prev_n_points = len(self._points_layer.data)
        self._points_layer.mode = "add"
        viewer.layers.selection.active = self._points_layer
        self._points_layer.events.data.connect(self._on_points_data_changed)

    def _stop_marking(self):
        """Exit marking mode (keeps the Points layer visible for reference)."""
        if not self._marking_active:
            return
        self._marking_active = False
        self.btn_start_marking.setText("Start marking")
        if self._points_layer is not None:
            try:
                self._points_layer.events.data.disconnect(self._on_points_data_changed)
                self._points_layer.mode = "pan_zoom"
            except Exception:
                pass

    def _on_points_data_changed(self, event=None):
        """Called when a point is added to the drift anchors layer."""
        layer = self._points_layer
        if layer is None:
            return
        n = len(layer.data)
        if n <= self._prev_n_points:
            return   # deletion or no change — ignore
        # A new point was added; use its (y, x) coordinates
        new_pt = layer.data[-1]   # shape: (z, y, x) because ndim=3
        x = float(new_pt[-1])
        y = float(new_pt[-2])
        p = self._current_plane()
        t = self._current_time()
        self._prev_n_points = n
        self._add_anchor(p, t, x, y)

    def _add_anchor(self, plane, t, x, y):
        """Add (t, x, y) as an anchor for the given plane and recompute shifts."""
        if plane not in self._anchors:
            self._anchors[plane] = []
        # Replace if same timepoint already exists for this plane
        self._anchors[plane] = [a for a in self._anchors[plane] if a[0] != t]
        self._anchors[plane].append((t, x, y))
        self._anchors[plane].sort(key=lambda a: a[0])
        self._refresh_anchor_list()
        if self.shift_model is not None:
            self._apply_anchor_interpolation(plane)

    def _apply_anchor_interpolation(self, plane):
        """Linearly interpolate shifts between anchors, then apply backwards
        exponential smoothing, and write into the shifts array.

        Backwards filtering (from t=T-1 to t=0) compensates for the user
        clicking an anchor slightly *after* the actual drift has occurred:
        the smoothed shift propagates the anchor's influence back in time
        with a time constant τ set by the tau_spinbox.
        """
        from scipy.interpolate import interp1d

        anchors = self._anchors.get(plane, [])
        if not anchors:
            return

        ref_t, ref_x, ref_y = anchors[0]   # first entry is the reference

        # Build control-point arrays.
        # dx = anchor_x - ref_x: positive when the feature moved right → shift mask right.
        ts  = np.array([a[0] for a in anchors], dtype=float)
        dxs = np.array([a[1] - ref_x for a in anchors], dtype=float)
        dys = np.array([a[2] - ref_y for a in anchors], dtype=float)

        all_t = np.arange(self.shift_model.n_timepoints, dtype=float)
        n = len(anchors)

        if n == 1:
            raw_dx = np.zeros(len(all_t))
            raw_dy = np.zeros(len(all_t))
        else:
            fx = interp1d(ts, dxs, kind="linear", bounds_error=False,
                          fill_value=(dxs[0], dxs[-1]))
            fy = interp1d(ts, dys, kind="linear", bounds_error=False,
                          fill_value=(dys[0], dys[-1]))
            raw_dx = fx(all_t)
            raw_dy = fy(all_t)

        # Backwards exponential smoothing
        tau_s  = self.tau_spinbox.value()
        acq_hz = self.acq_rate_spinbox.value()
        filt_dx = self._exp_backfilter(raw_dx, tau_s, acq_hz)
        filt_dy = self._exp_backfilter(raw_dy, tau_s, acq_hz)

        self.shift_model.shifts[:, plane, 0] = np.clip(
            np.round(filt_dx).astype(int), -32768, 32767
        )
        self.shift_model.shifts[:, plane, 1] = np.clip(
            np.round(filt_dy).astype(int), -32768, 32767
        )

        self._display_mask()
        self._update_heatmap()
        tau_msg = f", τ={tau_s:.2f} s" if tau_s > 0 else ""
        self.anchor_status.setText(
            f"plane {plane}: {n} anchor(s) — linear interp{tau_msg}"
        )

    @staticmethod
    def _exp_backfilter(arr, tau_s, acq_hz):
        """Backwards exponential smoothing with time constant *tau_s* seconds.

        Processes the array from t = T-1 down to t = 0 using:
            out[t] = α·x[t] + (1−α)·out[t+1],   α = 1 − exp(−dt/τ)

        This means a step change at t_click decays smoothly back toward
        earlier timepoints — compensating for the user clicking after the
        drift was actually visible.  τ = 0 returns an unchanged copy.
        """
        if tau_s <= 0.0 or acq_hz <= 0.0:
            return arr.copy()
        alpha = 1.0 - np.exp(-1.0 / (tau_s * acq_hz))
        out = arr.copy()
        for t in range(len(out) - 2, -1, -1):
            out[t] = alpha * arr[t] + (1.0 - alpha) * out[t + 1]
        return out

    def _refresh_anchor_list(self):
        """Rebuild the anchor list widget for the current plane."""
        self.anchor_list.clear()
        p = self._current_plane()
        anchors = self._anchors.get(p, [])
        if not anchors:
            return
        ref_t, ref_x, ref_y = anchors[0]
        for i, (t, x, y) in enumerate(anchors):
            if i == 0:
                label = f"t={t}  (reference)  x={x:.1f}  y={y:.1f}"
            else:
                dx = round(x - ref_x)
                dy = round(y - ref_y)
                label = f"t={t}  x={x:.1f}  y={y:.1f}  →  dx={dx}  dy={dy}"
            self.anchor_list.addItem(label)

    def _on_remove_anchor(self):
        """Remove the selected anchor and recompute interpolation."""
        p = self._current_plane()
        row = self.anchor_list.currentRow()
        if row < 0 or p not in self._anchors:
            return
        del self._anchors[p][row]
        self._refresh_anchor_list()
        if self.shift_model is not None:
            if self._anchors.get(p):
                self._apply_anchor_interpolation(p)
            else:
                # No anchors left — zero out this plane's shifts
                self.shift_model.shifts[:, p] = 0
                self._display_mask()
                self._update_heatmap()

    def _on_clear_plane_anchors(self):
        """Clear all anchors for the current plane, remove their points from
        the Points layer, and reset the plane's shifts to 0."""
        p = self._current_plane()
        self._anchors.pop(p, None)
        self.anchor_status.setText("")
        self._refresh_anchor_list()

        # Remove points belonging to this plane from the napari Points layer.
        # Points are (z, y, x) with ndim=3; z == current plane index.
        if self._points_layer is not None:
            try:
                data = self._points_layer.data
                if len(data) > 0:
                    keep = np.round(data[:, 0]).astype(int) != p
                    self._points_layer.data = data[keep]
                    self._prev_n_points = len(self._points_layer.data)
            except Exception:
                pass

        if self.shift_model is not None:
            self.shift_model.shifts[:, p] = 0
            self._display_mask()
            self._update_heatmap()

    def _on_tau_changed(self):
        """Re-run interpolation + filtering for every plane that has anchors."""
        if self.shift_model is None:
            return
        for plane, anchors in list(self._anchors.items()):
            if anchors:
                self._apply_anchor_interpolation(plane)

    def _display_current(self, auto_contrast=False):
        """Load and display the current timepoint's image and shifted mask."""
        stack_name = self.combo_stack.currentText()
        if not stack_name or self.shift_model is None:
            return

        t = self._current_time()
        img = self.data_manager.get_timepoint(stack_name, t)
        self.viewer_manager.show_image(img, name="reference", auto_contrast=auto_contrast)
        self._display_mask()
        self._update_time_marker()

    # ---- Label-edit sync ------------------------------------------------

    def _connect_label_events(self):
        """Connect to the napari labels layer's data event (once per layer)."""
        if self._label_event_connected:
            return
        try:
            layer = self.viewer_manager.viewer.layers["mask"]
            layer.events.data.connect(self._on_labels_edited)
            self._label_event_connected = True
        except KeyError:
            pass

    def _on_labels_edited(self, event=None):
        """Called by napari when the labels layer data changes."""
        if not self._our_update:
            self._pending_label_sync = True

    def _sync_user_edits(self):
        """Read back user edits from napari and store sparse per-(t, p) overrides.

        Only runs when _pending_label_sync is True (i.e., the user actually
        painted or erased since the last display). Only planes that differ
        from base_mask are stored, keeping memory usage minimal.
        """
        if not self._pending_label_sync or self.shift_model is None:
            return
        edited = self.viewer_manager.get_labels_data("mask")
        if edited is None:
            return
        t = self._current_time()
        for p in range(self.shift_model.n_planes):
            dx, dy = self.shift_model.get_shift(t, p)
            unshifted = self.shift_model._shift_plane(edited[p], -dx, -dy)
            if not np.array_equal(unshifted, self.shift_model.base_mask[p]):
                self.shift_model.set_plane_mask(t, p, unshifted)
            # If equal to base, leave any existing override in place rather than
            # silently clearing it (user may have painted then un-painted)
        self._pending_label_sync = False

    # ---- Mask display ---------------------------------------------------

    def _display_mask(self):
        """Apply shifts and display the mask for the current timepoint.

        Syncs any user edits first so they are preserved before overwriting.
        Skipped during playback: the sync reads the *previously displayed*
        labels and un-shifts them by the *new* timepoint's shift, which
        cancels the shift and makes the mask appear stuck at position 0.
        While an anchor is active, previews the anchored shift on the
        relevant plane(s) so the user can see where the mask will land.
        """
        if self._play_timer.isActive():
            self._pending_label_sync = False   # discard stale sync from previous frame
        else:
            self._sync_user_edits()

        if self.shift_model is None:
            return
        t = self._current_time()
        shifted = self.shift_model.apply_shifts_for_timepoint(t)

        self._our_update = True
        self.viewer_manager.show_labels(shifted, name="mask")
        self._our_update = False
        self._connect_label_events()

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
