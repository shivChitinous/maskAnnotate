import json

import numpy as np
import xarray as xr
from pathlib import Path


class DataManager:
    """Handles lazy loading of 4D stacks and 3D masks, caching, and saving."""

    def __init__(self):
        self.stacks = {}        # name -> xr.DataArray (lazy/chunked)
        self._agg_cache = {}    # (name, method) -> np.ndarray
        self.mask = None        # np.ndarray (planes, h, w)
        self.mask_path = None
        self.roi_sets = {}      # key name -> list of label IDs from JSON

    def load_stack(self, path):
        """Load a 4D .nc stack via xarray with chunked/dask access.

        Tries the default netcdf4 engine first, falls back to scipy for
        large CDF-2 files that netcdf4 can't handle.

        Returns (name, shape) tuple.
        """
        path = Path(path)
        kwargs = dict(
            decode_coords="coordinates",
            chunks={"volumes [s]": 1000, "planes [µm]": 1},
        )
        try:
            da = xr.open_dataarray(str(path), **kwargs)
        except Exception:
            da = xr.open_dataarray(str(path), engine="scipy", **kwargs)
        name = path.stem
        self.stacks[name] = da
        self._agg_cache = {
            k: v for k, v in self._agg_cache.items() if k[0] != name
        }
        return name, tuple(da.shape)

    def load_mask(self, path):
        """Load a 3D .npy mask. Returns shape tuple."""
        path = Path(path)
        self.mask = np.load(str(path)).astype(np.int32)
        self.mask_path = path
        return tuple(self.mask.shape)

    def load_roi_json(self, path):
        """Load ROI selection JSON. Returns dict of {set_name: [label_ids]}."""
        path = Path(path)
        with open(str(path)) as f:
            self.roi_sets = json.load(f)
        return self.roi_sets

    def get_stack_names(self):
        return list(self.stacks.keys())

    def get_aggregate(self, stack_name, method="mean"):
        """Compute and cache a time-aggregation of a 4D stack.

        method: 'mean', 'std', or 'max'
        Returns np.ndarray (planes, h, w).
        """
        key = (stack_name, method)
        if key in self._agg_cache:
            return self._agg_cache[key]

        da = self.stacks[stack_name]
        time_dim = da.dims[0]

        if method == "mean":
            result = da.mean(dim=time_dim).values
        elif method == "std":
            result = da.std(dim=time_dim).values
        elif method == "max":
            result = da.max(dim=time_dim).values
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        result = np.asarray(result, dtype=np.float32)
        self._agg_cache[key] = result
        return result

    def get_timepoint(self, stack_name, t):
        """Extract a single timepoint as np.ndarray (planes, h, w)."""
        da = self.stacks[stack_name]
        return np.asarray(da[t].values, dtype=np.float32)

    def get_n_timepoints(self, stack_name):
        da = self.stacks[stack_name]
        return da.shape[0]

    def get_n_planes(self, stack_name):
        da = self.stacks[stack_name]
        return da.shape[1]

    def save_mask_3d(self, path, mask):
        """Save a 3D mask as .npy."""
        np.save(str(path), mask)

    def save_mask_4d(self, path, shift_model, callback=None):
        """Generate and save the full 4D mask by applying shifts.

        callback(progress_fraction) is called during generation.
        """
        mask_4d = shift_model.generate_4d_mask(callback=callback)
        np.save(str(path), mask_4d)
        return mask_4d.shape
