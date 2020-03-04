#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    TriangulationParameters,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    pose_to_view_mat3x4,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4
)
import sortednp as snp

first_params = TriangulationParameters(1.0, 0, 0.1)
all_params = TriangulationParameters(1.0, 1, 0.1)


def try_add_points(point_cloud_builder, corner_storage, view_mats, intrinsic_mat, frame1, frame2, params):
    if frame1 == frame2 or view_mats[frame1] is None or view_mats[frame2] is None:
        return False
    correspondences = build_correspondences(corner_storage[frame1], corner_storage[frame2])
    points, points_ids, cos = triangulate_correspondences(correspondences, view_mats[frame1], view_mats[frame2], intrinsic_mat, params)
    point_cloud_builder.add_points(points_ids, points)
    return True


def try_set_view_matrix(point_cloud_builder, corner_storage, view_mats, intrinsic_mat, frame):
    if not view_mats[frame] is None:
        return False
    points = point_cloud_builder.points
    points_ids = point_cloud_builder.ids
    frame_corners = corner_storage[frame]
    _, (indices_1, indices_2) = snp.intersect(points_ids.flatten(), frame_corners.ids.flatten(), indices=True)
    object_points = points[indices_1]
    image_points = frame_corners.points[indices_2]
    if image_points.shape[0] < 5:
        return False
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, intrinsic_mat, np.zeros((4, 1)))
    except:
        return False
    if not success:
        return False
    view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
    return True


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frames_number = len(rgb_sequence)
    view_mats, point_cloud_builder = [None] * frames_number, PointCloudBuilder(np.array([]).astype(np.int64))

    frame1, pos1 = known_view_1
    frame2, pos2 = known_view_2
    view_mats[frame1] = pose_to_view_mat3x4(pos1)
    view_mats[frame2] = pose_to_view_mat3x4(pos2)

    try_add_points(point_cloud_builder, corner_storage, view_mats, intrinsic_mat, frame1, frame2, first_params)

    updated = True
    epoch = 0
    while updated:
        epoch += 1
        updated = False
        print(f"Current epoch: {epoch}")
        for frame in range(frames_number):
            try_set_view_matrix(point_cloud_builder, corner_storage, view_mats, intrinsic_mat, frame)
            for frame2 in range(frames_number):
                old_size = point_cloud_builder.points.shape[0]
                try_add_points(point_cloud_builder, corner_storage, view_mats, intrinsic_mat, frame, frame2, all_params)
                new_size = point_cloud_builder.points.shape[0]
                if old_size != new_size:
                    updated = True
            print(f'Processed {frame + 1} frames of {frames_number} in epoch {epoch}')

        views_unset = len(list(filter(lambda mat: mat is None, view_mats)))
        print(f'After epoch {epoch} {frames_number - views_unset} are set')

    unset_views_number = len(list(filter(lambda mat: mat is None, view_mats)))
    if unset_views_number != 0:
        raise ValueError(f"Failed to find view matrix for {unset_views_number} frames")

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
