import numpy as np

from detector.privacy import blur_people
from detector.types import Detection


def test_person_region_is_blurred_and_rest_untouched():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    out, n = blur_people(img, [Detection(50, 50, 100, 150, cls="person"), Detection(150, 0, 190, 40)], pad=0)
    assert n == 1
    assert out[50:150, 50:100].std() < img[50:150, 50:100].std() * 0.5   # blurred
    assert np.array_equal(out[:, 150:], img[:, 150:])                    # product area unchanged
