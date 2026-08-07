"""
Raw level arduino GPIO controll module
"""

from logging import getLogger
from queue import Queue
from threading import Condition, Lock, Thread
from typing import Callable, Optional

from serial import Serial

__all__ = ("PyDuino",)


def _dummy(*_):
    pass


class PyDuino:
    """
    Raw level arduino GPIO controll class
    """

    PACKET_START = 0x55
    PACKET_END = 0xAA

    PACKET_ID_INTERRUPT = 0
    PACKET_ID_MAX = 199
    PACKET_ID_ERROR = 0xFF

    def __init__(self, port: str, baudrate: int = 115200, *, debug: bool = False):
        self.debug = debug
        if self.debug:
            self.logger = getLogger("PYDUINO")
            self.logger.info("serial port %s (%d)", port, baudrate)

        self.__arduino = None
        self.__stop_requested = False  # to support graceful exit
        self.__force_stop = False
        self.__watcher = {}
        self.__values = [1] * 14  # Pin status
        if not port:
            raise ValueError("Serial port must be specified")

        # init arduino
        arduino = Serial(port=port, baudrate=baudrate, timeout=1)
        # Pulse DTR to ensure clean Arduino reset and synchronization
        try:
            arduino.dtr = False
            import time
            time.sleep(0.05)
            arduino.dtr = True
            time.sleep(0.2)
        except Exception:
            pass
        arduino.reset_input_buffer()
        arduino.reset_output_buffer()
        self.__arduino = arduino

        self.__tx_lock = Lock()
        self.__tx_cv = Condition()
        self.__tx_queue = Queue()
        self.__tx_packet_id = 1
        self.__tx = Thread(target=self.__transmitter, daemon=True)
        self.__tx.name = "Arduino Tx"
        self.__tx.start()

        self.__init_cv = Condition()
        self.__init_input_pins = (10, 11, 12, 13)
        self.__init_count = 0

        self.__rx = Thread(target=self.__receiver, daemon=True)
        self.__rx.name = "Arduino Rx"
        self.__rx.start()

        self.__init()

    def __del__(self):
        self.close()

    def __wait_for_input_status(self, pin: int, value: int, _):
        self.__values[pin] = value

        with self.__init_cv:
            self.__init_count += 1
            if self.__init_count >= len(self.__init_input_pins):
                self.__init_cv.notify_all()

    def __init(self):
        with self.__init_cv:
            self.__init_cv.wait(timeout=2.0)

        for idx in self.__init_input_pins:
            self.watch(idx)

    # Tx Thread
    def __transmitter(self):
        arduino = self.__arduino

        while True:
            packet = self.__tx_queue.get()
            if self.debug:
                self.logger.info(packet)

            pin, value, packet_id = packet
            self.__tx_queue.task_done()
            if pin is None:
                break  # End of Tx

            pin = int(pin) & 0xFF
            value = int(value) & 0xFF

            arduino.write(
                bytes([PyDuino.PACKET_START, packet_id, pin, value, PyDuino.PACKET_END])
            )

    # Rx Thread
    def __receiver(self):
        for idx in self.__init_input_pins:
            self.__values[idx] = 0xFF
            self.watch(idx, self.__wait_for_input_status)

        while self.__force_stop is False:
            packet = self.__receive_packet()
            if self.debug:
                self.logger.info(packet)
            if packet is None:
                continue

            # update pin status
            pin, value, packet_id = packet
            self.__values[pin] = value

            # run callback
            if packet_id == PyDuino.PACKET_ID_INTERRUPT:
                self.__watcher.get(pin, _dummy)(*packet)
            else:
                # TODO: check last packet id
                with self.__tx_cv:
                    self.__tx_cv.notify()

    def __receive_packet(self):
        arduino = self.__arduino

        data = arduino.read()
        if len(data) == 0 or data[0] != PyDuino.PACKET_START:
            return None

        packet = []
        while len(packet) != 4:
            data = arduino.read()
            if len(data) == 0:
                return None  # Timeout

            packet.append(data[0])

        packet_id, pin, value, packet_end = packet
        if packet_id < 0 or packet_id > PyDuino.PACKET_ID_ERROR:
            return None

        if pin < 0 or pin > len(self.__values):
            return None

        if packet_end != PyDuino.PACKET_END:
            return None

        return (pin, value, packet_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        """
        Close existing arduino connection
        """
        if self.__arduino is None:
            return

        self.__force_stop = True
        self.__stop_requested = True

        with self.__tx_cv:
            self.__tx_cv.notify_all()
        with self.__init_cv:
            self.__init_cv.notify_all()

        try:
            self.__tx_queue.put_nowait((None, None, None))
        except Exception:
            pass

        try:
            self.__arduino.close()
        except Exception:
            pass

        if self.debug:
            self.logger.info("stopped")

    def set(self, pin: int, value: int) -> bool:
        """
        Set arduino pin with given value
        """
        if self.__stop_requested:
            return False

        with self.__tx_lock:
            packet_id = self.__tx_packet_id

            with self.__tx_cv:
                self.__tx_queue.put((pin, value, packet_id))
                self.__tx_cv.wait(timeout=0.2)

            self.__tx_packet_id += 1
            if self.__tx_packet_id >= PyDuino.PACKET_ID_MAX:
                self.__tx_packet_id = 1

        return True

    def get(self, pin: int) -> int:
        """
        Get current arduino pin value from proxy
        """
        return self.__values[pin]

    def watch(
        self, pin: int, callback: Optional[Callable[[int, int, int], None]] = None
    ) -> None:
        """
        Set callback function for given pin
        """
        if callback is None:
            callback = _dummy

        self.__watcher[pin] = callback
