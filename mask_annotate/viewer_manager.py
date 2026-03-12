import numpy as np


class ViewerManager:
    """Manages napari viewer layers for the mask annotation workflow."""

    def __init__(self, viewer):
        self.viewer = viewer

    def show_image(self, data, name="reference", auto_contrast=False):
        """Add or update an Image layer.

        If auto_contrast is True, sets contrast limits to data min/max.
        """
        try:
            layer = self.viewer.layers[name]
            layer.data = data
        except KeyError:
            layer = self.viewer.add_image(data, name=name)
            auto_contrast = True  # always auto-contrast on first add
        if auto_contrast:
            layer.contrast_limits = (float(np.nanmin(data)), float(np.nanmax(data)))

    def show_labels(self, mask, name="mask"):
        """Add or update a Labels layer."""
        mask = np.asarray(mask, dtype=np.int32)
        try:
            layer = self.viewer.layers[name]
            layer.data = mask
        except KeyError:
            self.viewer.add_labels(mask, name=name)

    def get_labels_data(self, name="mask"):
        """Retrieve the current labels data (reflecting any user edits)."""
        try:
            return np.asarray(self.viewer.layers[name].data, dtype=np.int32)
        except KeyError:
            return None

    def remove_layer(self, name):
        """Remove a layer by name if it exists."""
        try:
            self.viewer.layers.remove(name)
        except ValueError:
            pass

    def clear(self):
        """Remove all layers."""
        self.viewer.layers.clear()

    def get_current_plane(self):
        """Get the currently displayed plane index from napari dims."""
        if self.viewer.dims.ndim >= 1:
            return int(self.viewer.dims.current_step[0])
        return 0
