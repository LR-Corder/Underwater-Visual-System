# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

import time
from pathlib import Path

import cv2

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_SAVE_PATH = Path('output.avi')
DEFAULT_FPS = 20.0
DEFAULT_DUR = 10  # seconds
DEFAULT_RES = (1280, 480)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def record_video(save_path: str | Path = DEFAULT_SAVE_PATH,
                 duration: float = DEFAULT_DUR,
                 fps: float = DEFAULT_FPS,
                 resolution: tuple[int, int] = DEFAULT_RES) -> None:
    """Record video from the default webcam.

    Args:
        save_path: Output file path.
        duration:  Recording length in seconds.
        fps:       Frames per second.
        resolution: (width, height).
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera.')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(save_path), fourcc, fps, resolution)

    frames_to_record = int(duration * fps)
    frame_count = 0

    try:
        while cap.isOpened() and frame_count < frames_to_record:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break

            # Optional processing: e.g., cv2.flip(frame, 0)
            out.write(frame)

            # Display feedback (optional)
            # cv2.imshow('recording', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            elapsed = time.perf_counter() - t0
            print(f'Frame {frame_count + 1}/{frames_to_record}, '
                  f'elapsed: {elapsed:.3f}s')
            frame_count += 1
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    record_video()