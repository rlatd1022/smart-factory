#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
from openvino import Core

from iotdemo import FactoryController, MotionDetector, ColorDetector

FORCE_STOP = False


def thread_cam1(q):
    # TODO: MotionDetector
    detector = MotionDetector()
    detector.load_preset("motion.cfg", "default")

    # TODO: Load and initialize OpenVINO
    core = Core()
    model = core.read_model("resources/model.xml")
    compiled_model = core.compile_model(model, "CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # TODO: Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture("resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))

        # TODO: Motion detect
        detected = detector.detect(frame)
        if detected is None:
            continue

        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(("VIDEO:Cam1 detected", detected))

        # abnormal detect
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        reshaped = detected[:, :, [2, 1, 0]]
        np_data = np.moveaxis(reshaped, -1, 0)
        preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        batch_tensor = np.stack(preprocessed_numpy, axis=0)

        # TODO: Inference OpenVINO
        results = compiled_model([batch_tensor])[output_layer]

        # TODO: Calculate ratios
        x_ratio = results[0][0] * 100
        circle_ratio = results[0][1] * 100
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # TODO: in queue for moving the actuator 1
        if x_ratio > 50:
            q.put(("PUSH", 1))

    cap.release()
    q.put(("DONE", None))
    exit()


def thread_cam2(q):
    # TODO: MotionDetector
    motion_detector = MotionDetector()
    motion_detector.load_preset("motion.cfg", "default")

    # TODO: ColorDetector
    color_detector = ColorDetector()
    color_detector.load_preset("color.cfg", "default")

    # TODO: Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live", frame))

        # TODO: Detect motion
        detected = motion_detector.detect(frame)
        if detected is None:
            continue

        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(("VIDEO:Cam2 detected", detected))

        # TODO: Detect color
        results = color_detector.detect(detected)
        if len(results) > 0:
            name, ratio = results[0]
        else:
            continue

        # TODO: Compute ratio
        ratio = ratio * 100
        print(f"{name}: {ratio:.2f}%")

        # TODO: Enqueue to handle actuator 2
        if ratio > 30:
            q.put(("PUSH", 2))

    cap.release()
    q.put(("DONE", None))
    exit()


def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    global FORCE_STOP

    parser = ArgumentParser(prog="python3 factory.py", description="Factory tool")

    parser.add_argument("-d", "--device", default=None, type=str, help="Arduino port")
    args = parser.parse_args()

    # TODO: Create a Queue
    q = Queue()

    # TODO: Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()

    with FactoryController(
        conn=FactoryController.Connector.ARDUINO, port=args.device, debug=True
    ) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

            # TODO: get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            try:
                name, data = q.get(timeout=0.1)
            except Empty:
                continue

            # TODO: show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            if name.startswith("VIDEO:"):
                imshow(name[6:], data)

            # TODO: Control actuator, name == 'PUSH'
            elif name == "PUSH":
                print(f"Kick actuator {data}")
                ctrl.push_actuator(data)

            if name == "DONE":
                FORCE_STOP = True

            q.task_done()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        os._exit(0)
