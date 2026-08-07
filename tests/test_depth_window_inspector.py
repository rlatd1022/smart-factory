import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from iotdemo.depth import DepthWindowInspector
from iotdemo.factory_controller.factory_controller import FactoryController
from iotdemo.factory_controller.pins import Outputs


class DepthWindowInspectorTest(unittest.TestCase):
    def setUp(self):
        self.inspector = DepthWindowInspector(
            ref_depth_mm=388.0,
            tol_mm=1.0,
            kicker_num=2,
            min_samples=3,
            idle_debounce_frames=2,
        )

    def test_good_part_minimum_388mm_no_kick(self):
        """388mm가 최소값인 경우 양품(OK)으로 판정하고 키커를 동작시키지 않음."""
        # Sequence of depth measurements during sample transit
        measurements = [400.0, 395.0, 388.0, 392.0, 400.0]
        for d in measurements:
            res = self.inspector.process_depth(d)
            self.assertEqual(res.event, "MEASURING")
            self.assertEqual(res.status, "MEASURE")
            self.assertFalse(res.should_kick)

        # Transition to N/A (sample passed)
        res1 = self.inspector.process_depth(None)  # debounce frame 1
        res2 = self.inspector.process_depth(None)  # debounce frame 2 -> OBJECT_DONE

        self.assertEqual(res2.event, "OBJECT_DONE")
        self.assertEqual(res2.status, "OK")
        self.assertTrue(res2.ok)
        self.assertAlmostEqual(res2.min_depth_mm, 388.0)
        self.assertFalse(res2.should_kick, "Good part must NOT trigger kicker")

    def test_defect_part_minimum_390mm_triggers_kick(self):
        """390mm가 최소값인 경우 불량(NG)으로 판정하고 키커를 동작시킴."""
        measurements = [405.0, 398.0, 390.0, 395.0]
        for d in measurements:
            self.inspector.process_depth(d)

        # Transition to N/A
        self.inspector.process_depth(None)
        res = self.inspector.process_depth(None)

        self.assertEqual(res.event, "OBJECT_DONE")
        self.assertEqual(res.status, "NG")
        self.assertFalse(res.ok)
        self.assertAlmostEqual(res.min_depth_mm, 390.0)
        self.assertTrue(res.should_kick, "Defective part (390mm) MUST trigger kicker")
        self.assertEqual(res.kicker_num, 2)

    def test_defect_part_minimum_380mm_triggers_kick(self):
        """380mm가 최소값인 경우 불량(NG)으로 판정하고 키커를 동작시킴."""
        measurements = [390.0, 385.0, 380.0, 385.0]
        for d in measurements:
            self.inspector.process_depth(d)

        self.inspector.process_depth(None)
        res = self.inspector.process_depth(None)

        self.assertEqual(res.event, "OBJECT_DONE")
        self.assertEqual(res.status, "NG")
        self.assertFalse(res.ok)
        self.assertAlmostEqual(res.min_depth_mm, 380.0)
        self.assertTrue(res.should_kick)

    def test_glitch_noise_ignored(self):
        """최소 샘플 수 미만(1개)의 노이즈는 무시."""
        self.inspector.process_depth(388.0)
        self.inspector.process_depth(None)
        res = self.inspector.process_depth(None)
        self.assertEqual(res.event, "IDLE")
        self.assertFalse(res.should_kick)

    def test_process_frame_center_mask_detection(self):
        """화면 정중앙에 파란색 샘플이 있을 때만 마스크 게이팅 통과 및 뎁스 측정."""
        h, w = 480, 640
        # Create background (black/empty)
        rgb_empty = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.full((h, w), 388, dtype=np.uint16)

        # Empty frame -> should be EMPTY
        res = self.inspector.process_frame(depth, rgb_empty)
        self.assertEqual(res.status, "EMPTY")
        self.assertFalse(res.mask_ok)

        # Draw blue circular sample at center (w//2, h//2) = (320, 240)
        # HSV: (120, 200, 200) -> BGR: roughly (200, 100, 50)
        rgb_sample = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(rgb_sample, (320, 240), 25, (200, 100, 50), -1)

        # Sample at center -> MEASURING
        res = self.inspector.process_frame(depth, rgb_sample)
        self.assertEqual(res.status, "MEASURE")
        self.assertTrue(res.mask_ok)
        self.assertAlmostEqual(res.current_depth_mm, 388.0)


class FactoryControllerActuatorTest(unittest.TestCase):
    @patch("iotdemo.factory_controller.factory_controller.PyDuino")
    def test_reversed_actuator_mapping(self, mock_pyduino):
        ctrl = FactoryController(
            conn=FactoryController.Connector.ARDUINO,
            port="/dev/dummy",
            debug=False,
            pulse_duration=0.05,
            reverse_actuator=True,
        )
        ctrl._FactoryController__device_name = "arduino"
        ctrl._FactoryController__device = MagicMock()

        # push_actuator(1) -> should map to ACTUATOR_2 (Pin 7)
        ctrl.push_actuator(1)
        ctrl._FactoryController__device.set.assert_any_call(Outputs.ACTUATOR_2, False)

        ctrl._FactoryController__device.reset_mock()

        # push_actuator(2) -> should map to ACTUATOR_1 (Pin 6)
        ctrl.push_actuator(2)
        ctrl._FactoryController__device.set.assert_any_call(Outputs.ACTUATOR_1, False)

        self.assertEqual(ctrl.pulse_duration, 0.05)


if __name__ == "__main__":
    unittest.main()
