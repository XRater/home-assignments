#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli

MAX_CORNERS = 1000
MIN_DISTANCE = 15
MAX_ERROR = 6
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners=MAX_CORNERS,
                      qualityLevel=0.05,
                      minDistance=MIN_DISTANCE,
                      blockSize=MIN_DISTANCE)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def build_frame_corners(corners, ids, next_id) -> FrameCorners:
    corners_number = corners.shape[0]
    new_corners = corners.shape[0] - len(ids)
    next_id1 = next_id + new_corners
    new_ids = np.concatenate((ids, np.arange(next_id, next_id1)))
    return FrameCorners(
        new_ids,
        np.array(corners),
        np.ones(corners_number) * 5
    ), new_ids, next_id1


def add_new_coreners(corners_old, corners_new):
    corners = corners_old
    for corner in corners_new:
        # if corners.shape[0] > MAX_CORNERS:
        #     continue
        bad = False
        for corner_old in corners:
            if ((corner_old - corner) ** 2).sum() < MIN_DISTANCE ** 2:
                bad = True
                break
        if not bad:
            corners = np.vstack([corners, [corner]])
    return corners


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    image_0 = (image_0 * 256).astype(np.uint8)
    corners_cv = cv2.goodFeaturesToTrack(image_0, **feature_params)
    corners, ids, next_id = build_frame_corners(corners_cv, np.array([]).astype(np.int32), 0)
    builder.set_corners_at_frame(0, corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        corners_new = cv2.goodFeaturesToTrack(image_1, **feature_params)
        image_1 = (image_1 * 256).astype(np.uint8)
        corners_cv1, _, err = cv2.calcOpticalFlowPyrLK(image_0, image_1, corners_cv, None, **lk_params)

        corners_cv_filtered = corners_cv1[np.stack([err, err]).transpose((1,2,0)) < MAX_ERROR].reshape(-1, 1, 2)
        ids = ids[err.squeeze() < MAX_ERROR]

        coreners_cv1 = add_new_coreners(corners_cv_filtered, corners_new)
        corners, ids, next_id = build_frame_corners(coreners_cv1, ids, next_id)
        builder.set_corners_at_frame(frame, corners)

        image_0 = image_1
        corners_cv = coreners_cv1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
