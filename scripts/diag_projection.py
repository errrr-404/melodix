import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tests.helpers import blank_page, draw_staff

from melodix.geometry.deskew import (
    DEFAULT_CONFIG,
    _working_mask,
    estimate_skew_projection,
    rotate_image,
)
from melodix.geometry.staff import binarize, to_grayscale

page = blank_page()
page2 = draw_staff(page, top_row=200, spacing=14)
print('before draw unique', np.unique(page2)[:10])
rot = rotate_image(page2, 2.0, border_value=255)
print('rot uniq', np.unique(rot)[:10])
gray = to_grayscale(rot)
print('gray uniq', np.unique(gray)[:10])
binmask = binarize(gray)
print('bin uniq', np.unique(binmask))
mask = _working_mask(rot, DEFAULT_CONFIG)
print('mask uniq', np.unique(mask), 'ink', int((mask>0).sum()))
print('estimate', estimate_skew_projection(rot))
