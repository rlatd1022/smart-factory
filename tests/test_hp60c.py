import unittest

import numpy as np

from iotdemo.depth.hp60c import decode_depth_frame, depth_at


class HP60CTest(unittest.TestCase):
    def test_decode_uint16(self):
        source = np.array([[200, 1234], [4000, 0]], dtype=np.uint16)
        np.testing.assert_array_equal(decode_depth_frame(source, width=2), source)

    def test_decode_packed_bytes(self):
        source = np.array([[200, 1234], [4000, 0]], dtype="<u2")
        packed = source.view(np.uint8).reshape(2, 2, 2)
        np.testing.assert_array_equal(decode_depth_frame(packed, width=2), source)

    def test_depth_at_uses_valid_median(self):
        source = np.array(
            [[0, 0, 0], [999, 1000, 1001], [65000, 0, 0]], dtype=np.uint16
        )
        self.assertAlmostEqual(depth_at(source, 1, 1, radius=1), 1.0)

    def test_depth_at_rejects_invalid_area(self):
        source = np.zeros((3, 3), dtype=np.uint16)
        self.assertIsNone(depth_at(source, 1, 1))


if __name__ == "__main__":
    unittest.main()
