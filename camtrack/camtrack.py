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

    view_mats, point_cloud_builder = [], PointCloudBuilder(np.array([]).astype(np.int64))

    frame1, pos1 = known_view_1
    frame2, pos2 = known_view_2
    view_mat1 = pose_to_view_mat3x4(pos1)
    view_mat2 = pose_to_view_mat3x4(pos2)

    correspondences = build_correspondences(corner_storage[frame1], corner_storage[frame2])
    points, points_ids, cos = triangulate_correspondences(correspondences, view_mat1, view_mat2, intrinsic_mat, TriangulationParameters(1, 0, 0.1))
    point_cloud_builder.add_points(points_ids, points)

    for frame in range(len(rgb_sequence)):
        points = point_cloud_builder.points
        points_ids = point_cloud_builder.ids
        frameCorners = corner_storage[frame]
        _, (indices_1, indices_2) = snp.intersect(points_ids.flatten(), frameCorners.ids.flatten(), indices=True)
        success, rvec, tvec, inliers = \
            cv2.solvePnPRansac(points[indices_1], frameCorners.points[indices_2], intrinsic_mat, np.zeros((4, 1)))
        view_mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
        view_mats.append(view_mat)

        for f in range(frame):
            correspondences_new = build_correspondences(corner_storage[f], corner_storage[frame])

            points_new, points_ids_new, _ = triangulate_correspondences(correspondences_new, view_mats[f], view_mats[frame], intrinsic_mat, TriangulationParameters(1, 1, 0.1))
            # old_size = point_cloud_builder.points.shape[0]
            point_cloud_builder.add_points(points_ids_new, points_new)
            # new_size = point_cloud_builder.points.shape[0]
            # if old_size != new_size:
            #     print(f"Added {new_size - old_size}")


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
