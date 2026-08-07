"""
Smart Factory HW module controller
"""

from logging import getLogger
from os import listdir
from sys import platform
from time import sleep
from typing import Optional, Union
from enum import Enum

from iotdemo.common.debounce import debounce
from iotdemo.factory_controller.pins import Inputs, Outputs
from iotdemo.factory_controller.pyduino import PyDuino
from iotdemo.factory_controller.pyft232 import PyFt232

__all__ = ("FactoryController",)


class FactoryController:
    """
    Factory HW controller class
    """

    DEV_ON = False
    DEV_OFF = True

    class Connector(Enum):
        FT232 = 1
        ARDUINO = 2
        AUTO = 3

    def __init__(
        self,
        conn: Connector,
        port: Optional[Union[str, int]] = None,
        *,
        debug: bool = True,
        pulse_duration: float = 0.05,
        reverse_actuator: bool = True,
    ):
        self.debug = debug
        self.pulse_duration = pulse_duration
        self.reverse_actuator = reverse_actuator
        if self.debug:
            self.logger = getLogger("CONTROLLER")

        self.__force_stop = False

        self.__device = None
        self.__device_name = None

        # open device
        candidate_ports = self.__get_candidate_ports(port if port is not None else -1)
        self.port = None

        for candidate in candidate_ports:
            if conn == FactoryController.Connector.FT232 and "ttyUSB" in candidate:
                try:
                    self.__device = PyFt232(candidate, debug=debug)
                    self.__device_name = "ft232"
                    self.port = candidate
                    break
                except Exception as e:
                    if self.debug:
                        self.logger.warning(f"FT232 connection failed on {candidate}: {e}")
                    self.__device = None
            else:
                try:
                    self.__device = PyDuino(candidate, debug=debug)
                    self.__device_name = "arduino"
                    self.port = candidate
                    break
                except Exception as e:
                    if self.debug:
                        self.logger.warning(f"PyDuino connection failed on {candidate}: {e}")
                    self.__device = None

        if self.__device_name == "arduino" and not self.is_dummy:
            # Arduino beacon indicator
            self.red = False
            self.orange = True
            self.green = False

            self.defect_sensor_status = False
            self.color_sensor_status = False

            # interrupt handler
            self.__device.watch(Inputs.START_BUTTON, self.__button_interrupt)
            self.__device.watch(Inputs.STOP_BUTTON, self.__button_interrupt)

            self.__device.watch(
                Inputs.PHOTOELECTRIC_SENSOR_1, self.__sensor_interrupt
            )
            self.__device.watch(
                Inputs.PHOTOELECTRIC_SENSOR_2, self.__sensor_interrupt
            )

        if self.debug:
            self.logger.info(
                f'use {"Dummy" if self.is_dummy else f"Arduino ({self.port})"} Controller'
            )

    def __get_candidate_ports(self, port) -> list:
        if isinstance(port, str):
            return [port]

        if not isinstance(port, int):
            raise RuntimeError(f"Invalid port argument type: {port} - {type(port)}")

        if platform in {"win32", "cygwin"}:
            return [f"COM{port}"] if port != -1 else [f"COM{i}" for i in range(1, 10)]

        if not platform.startswith("linux"):
            raise RuntimeError("Not supported OS")

        if port == -1:
            candidates = []
            try:
                import os
                raw_list = sorted([p for p in listdir("/dev") if p.startswith("ttyACM") or p.startswith("ttyUSB")])
                for p in raw_list:
                    dev_path = f"/dev/{p}"
                    # Prioritize readable and writable devices
                    if os.access(dev_path, os.R_OK | os.W_OK):
                        candidates.append(dev_path)
                # If no accessible devices found by os.access, append all raw candidates as fallback
                if not candidates:
                    candidates = [f"/dev/{p}" for p in raw_list]
            except Exception:
                pass
            return candidates
        else:
            return [f"/dev/ttyACM{port}"]

    def __del__(self):
        self.close()

    def __enter__(self):
        if not self.is_dummy:
            self.system_start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_dummy:
            self.system_stop()
        self.close()

    def __set(self, pin, value):
        if self.is_dummy or self.__force_stop:
            return

        self.__device.set(pin, value)

    def __get(self, pin):
        if self.is_dummy:
            return None

        return self.__device.get(pin)

    def __led(self, pin, on):
        if self.__device_name == "arduino":
            self.__set(
                pin, FactoryController.DEV_ON if on else FactoryController.DEV_OFF
            )

    def __actuator(self, actuator_id, duration: Optional[float] = None):
        d = duration if duration is not None else self.pulse_duration
        if self.__device_name == "arduino":
            self.__set(actuator_id, FactoryController.DEV_ON)
            sleep(d)
            self.__set(actuator_id, FactoryController.DEV_OFF)
        elif self.__device_name == "ft232":
            self.__set(PyFt232.PKT_CMD_DETECTION, actuator_id)
            sleep(d * 5)
            self.__set(PyFt232.PKT_CMD_DETECTION, PyFt232.PKT_CMD_DETECTION_0)

    @debounce(0.1)
    def __button_interrupt(self, pin: int, value: int, *_):
        if value != 0 or self.__force_stop:
            return

        if pin == Inputs.START_BUTTON:
            self.system_start()

        elif pin == Inputs.STOP_BUTTON:
            self.system_stop()

    @debounce(0.3)
    def __sensor_interrupt(self, pin: int, value: int, *_):
        status = value == 0

        if pin == Inputs.PHOTOELECTRIC_SENSOR_1:
            self.defect_sensor_status = status

        elif pin == Inputs.PHOTOELECTRIC_SENSOR_2:
            self.color_sensor_status = status

    ###########################################################################
    # Public properties

    @property
    def is_dummy(self):
        return self.__device is None

    @property
    def red(self):
        return self.__get(Outputs.BEACON_RED) == FactoryController.DEV_ON

    @red.setter
    def red(self, on):
        self.__led(Outputs.BEACON_RED, on)

    @property
    def orange(self):
        return self.__get(Outputs.BEACON_ORANGE) == FactoryController.DEV_ON

    @orange.setter
    def orange(self, on):
        self.__led(Outputs.BEACON_ORANGE, on)

    @property
    def green(self):
        return self.__get(Outputs.BEACON_GREEN) == FactoryController.DEV_ON

    @green.setter
    def green(self, on):
        self.__led(Outputs.BEACON_GREEN, on)

    @property
    def conveyor(self):
        return self.__get(Outputs.CONVEYOR_EN) == FactoryController.DEV_ON

    @conveyor.setter
    def conveyor(self, on):
        if on:
            self.__set(Outputs.CONVEYOR_EN, FactoryController.DEV_ON)
            speed = getattr(self, "_current_speed_pwm", 255)
            self.__set(Outputs.CONVEYOR_PWM, speed)
        else:
            self.__set(Outputs.CONVEYOR_PWM, 0)
            self.__set(Outputs.CONVEYOR_EN, FactoryController.DEV_OFF)

    @property
    def conveyor_speed(self) -> int:
        """컨베이어 PWM 속도 (0~255)."""
        return getattr(self, "_current_speed_pwm", 255)

    @conveyor_speed.setter
    def conveyor_speed(self, pwm_val: int):
        """컨베이어 PWM 속도 조절 (0~255)."""
        val = max(0, min(255, int(pwm_val)))
        self._current_speed_pwm = val
        if not self.is_dummy:
            self.__set(Outputs.CONVEYOR_PWM, val)
        if self.debug and hasattr(self, "logger"):
            self.logger.info(f"Conveyor Speed set to {val} PWM ({val*100//255}%)")

    def set_conveyor_speed(self, speed_pwm: int) -> None:
        """컨베이어 속도 조절 헬퍼."""
        self.conveyor_speed = speed_pwm

    def conveyor_stop(self) -> None:
        """컨베이어 일시 정지 (PWM 0 및 EN OFF)."""
        self.conveyor = False

    def conveyor_start(self, speed: Optional[int] = None) -> None:
        """컨베이어 가동 (속도 지정 가능)."""
        if speed is not None:
            self._current_speed_pwm = max(0, min(255, int(speed)))
        self.conveyor = True

    def manual_kick(self, kicker_num: int) -> None:
        """수동 키커 배출 테스트 트리거."""
        self.push_actuator(kicker_num)

    @property
    def is_running(self) -> bool:
        """공정 작동 중 여부."""
        return self.green and self.conveyor

    def push_actuator(self, num):
        if self.reverse_actuator:
            # 반전: 1차 검사(비전) -> ACTUATOR_2, 2차 검사(뎁스) -> ACTUATOR_1
            act_id = Outputs.ACTUATOR_2 if num == 1 else Outputs.ACTUATOR_1
        else:
            act_id = Outputs.ACTUATOR_1 if num == 1 else Outputs.ACTUATOR_2

        if self.__device_name == "arduino":
            self.__actuator(act_id)
        elif self.__device_name == "ft232":
            mapped_num = (
                2 if (self.reverse_actuator and num == 1)
                else (1 if (self.reverse_actuator and num == 2) else num)
            )
            self.__actuator(mapped_num)

    ###########################################################################
    # Public methods

    def system_start(self) -> None:
        if self.debug:
            self.logger.info("Start System")

        if self.__device_name == "ft232":
            self.__set(PyFt232.PKT_CMD_START, PyFt232.PKT_CMD_START_START)
            self.__set(PyFt232.PKT_CMD_SPEED, PyFt232.PKT_CMD_SPEED_UP)
            self.__set(PyFt232.PKT_CMD_SPEED, PyFt232.PKT_CMD_SPEED_UP)
            self.__set(PyFt232.PKT_CMD_SPEED, PyFt232.PKT_CMD_SPEED_UP)
            self.__set(PyFt232.PKT_CMD_SPEED, PyFt232.PKT_CMD_SPEED_UP)
        else:
            self.red = False
            self.green = True
            self.conveyor = True

    def system_stop(self) -> None:
        if self.debug:
            self.logger.info("Stop System")

        if self.__device_name == "ft232":
            self.__set(PyFt232.PKT_CMD_START, PyFt232.PKT_CMD_START_STOP)
        else:
            self.red = True
            self.green = False
            self.conveyor = False

    def close(self):
        if self.debug:
            self.logger.info("Shutdown System")

        self.__force_stop = True
        if self.is_dummy:
            return

        self.__device.close()
