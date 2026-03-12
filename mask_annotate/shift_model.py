import numpy as np
from scipy.ndimage import shift as ndi_shift


class ShiftModel:
    """Stores per-(timepoint, plane) x/y shifts and applies them to a base mask.

    Shifts are integer pixel displacements. Pixels that move out of bounds
    are discarded (set to 0/background) — no wrapping.
    """

    def __init__(self, base_mask, n_timepoints):
        """
        Parameters
        ----------
        base_mask : np.ndarray, shape (n_planes, h, w)
            The cleaned 3D mask from Stage 2.
        n_timepoints : int
            Number of time points in the 4D stack.
        """
        self.base_mask = np.asarray(base_mask, dtype=np.int32)
        self.n_timepoints = n_timepoints
        self.n_planes = base_mask.shape[0]
        # shifts[t, p] = (dx, dy) as integer pixel shifts
        self.shifts = np.zeros((n_timepoints, self.n_planes, 2), dtype=np.int16)

    def set_shift(self, t, plane, dx, dy):
        self.shifts[t, plane] = [dx, dy]

    def get_shift(self, t, plane):
        """Returns (dx, dy) for a given timepoint and plane."""
        dx, dy = self.shifts[t, plane]
        return int(dx), int(dy)

    def set_shift_all_timepoints(self, plane, dx, dy):
        """Copy a shift to all timepoints for a given plane."""
        self.shifts[:, plane] = [dx, dy]

    def set_shift_all_planes(self, t, dx, dy):
        """Copy a shift to all planes for a given timepoint."""
        self.shifts[t, :] = [dx, dy]

    def set_shift_range(self, t_start, t_end, plane, dx, dy):
        """Apply (dx, dy) to timepoints t_start..t_end (inclusive) for one plane."""
        self.shifts[t_start:t_end + 1, plane] = [dx, dy]

    def reset_all(self):
        """Zero out all shifts."""
        self.shifts[:] = 0

    def apply_shifts_for_timepoint(self, t):
        """Apply stored shifts to the base mask for a single timepoint.

        Returns np.ndarray of shape (n_planes, h, w).
        """
        result = np.empty_like(self.base_mask)
        for p in range(self.n_planes):
            dx, dy = self.get_shift(t, p)
            result[p] = self._shift_plane(self.base_mask[p], dx, dy)
        return result

    @staticmethod
    def _shift_plane(plane_mask, dx, dy):
        """Shift a 2D mask plane by (dx, dy) pixels.

        Uses nearest-neighbor interpolation (order=0) to preserve integer labels.
        mode='constant' with cval=0 ensures out-of-bounds pixels become background,
        effectively clipping or eliminating ROIs that leave the field of view.
        """
        if dx == 0 and dy == 0:
            return plane_mask.copy()
        # scipy.ndimage.shift takes (row_shift, col_shift) = (dy, dx)
        return ndi_shift(
            plane_mask, shift=(dy, dx), order=0, mode="constant", cval=0
        ).astype(plane_mask.dtype)

    def generate_4d_mask(self, callback=None):
        """Materialize the full 4D shifted mask.

        Parameters
        ----------
        callback : callable, optional
            Called with progress fraction (0.0 to 1.0) during generation.

        Returns
        -------
        np.ndarray of shape (n_timepoints, n_planes, h, w)
        """
        h, w = self.base_mask.shape[1], self.base_mask.shape[2]
        mask_4d = np.empty(
            (self.n_timepoints, self.n_planes, h, w), dtype=self.base_mask.dtype
        )
        for t in range(self.n_timepoints):
            mask_4d[t] = self.apply_shifts_for_timepoint(t)
            if callback is not None:
                callback((t + 1) / self.n_timepoints)
        return mask_4d
