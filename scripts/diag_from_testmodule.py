import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tests.test_deskew as td

from melodix.geometry.deskew import estimate_skew_projection

for angle in [-6.0, -3.5, -0.8, 0.0, 0.8, 2.0, 3.5, 6.0]:
    page = td.tilted_page(angle)
    est = estimate_skew_projection(page)
    print('angle', angle, '-> estimate', est)
