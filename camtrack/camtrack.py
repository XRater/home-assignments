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
    rodrigues_and_translation_to_view_mat3x4,
    eye3x4
)
import sortednp as snp
from bundle_adjustment import run_bundle_adjustment

params = TriangulationParameters(2.2, 3, 0.1)


def try_add_points(point_cloud_builder, corner_storage, view_mats, intrinsic_mat, frame1, frame2, params):
    if frame1 == frame2 or view_mats[frame1] is None or view_mats[frame2] is None:
        return False
    correspondences = build_correspondences(corner_storage[frame1], corner_storage[frame2])
    points, points_ids, cos = triangulate_correspondences(correspondences, view_mats[frame1], view_mats[frame2], intrinsic_mat, params)
    point_cloud_builder.add_points(points_ids, points)
    return True


def estimate_views(corner_storage, intrinsic_mat, pose1, pose2, frame1, frame2):
    view_mat_1 = pose_to_view_mat3x4(pose1)
    view_mat_2 = pose_to_view_mat3x4(pose2)
    correspondences = build_correspondences(corner_storage[frame1], corner_storage[frame2])
    points, points_ids, cos = triangulate_correspondences(correspondences, view_mat_1, view_mat_2, intrinsic_mat, params)
    return points.shape[0], cos


def remove_outliers(point_cloud_builder, ids, inliers):
    filtered_number = 0
    for id in ids:
        if not id in inliers:
            point_cloud_builder.remove_point(id)
            filtered_number += 1
    print(f"Filtered {filtered_number} outliers")


def try_set_view_matrix(point_cloud_builder, corner_storage, view_mats, inliers_array, intrinsic_mat, frame):
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
    if inliers_array[frame] is None or inliers_array[frame] < len(inliers):
        print(f'Matrix was updated for frame {frame} with {len(inliers)} inliers')
        view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
        inliers_array[frame] = len(inliers)
        return True
    return False


def generate_frame_pairs_with_fixed_init(max_frame):
    init = [0, max_frame // 3, max_frame // 2, max_frame - 1]
    pairs = []
    for i in init:
        for j in range(0, max_frame):
            if not j == i:
                pairs.append((i, j))
    return pairs


def generate_frame_pairs_all(max_frame):
    pairs = []
    for i in range(max_frame):
        for j in range(i, max_frame):
            pairs.append((i, j))
    return pairs


def get_views_by_frames(corner_storage, intrinsic_mat, frame1, frame2):
    correspondences = build_correspondences(corner_storage[frame1], corner_storage[frame2])
    if len(correspondences[0]) < 5:
        print(f"Not enouth points to build essential matrix for frames {frame1}, {frame2}")
        return False, None, None
    points1, points2 = correspondences[1], correspondences[2]
    essential_matrix, mask = cv2.findEssentialMat(points1, points2, cameraMatrix=intrinsic_mat)
    if essential_matrix is None or essential_matrix.shape != (3, 3):
        return False, None, None
    points_number, R, t, mask = cv2.recoverPose(essential_matrix, points1, points2, intrinsic_mat)
    first_pos = view_mat3x4_to_pose(np.eye(3, 4))
    second_pos = view_mat3x4_to_pose(np.hstack([R, t]))
    if points_number < 10:
        return False, None, None
    return True, first_pos, second_pos


def get_best_result(results):
    best = None
    for res in results:
        if best is None:
            best = res
            continue
        _, _, (points_number, cos) = res
        _, _, (best_points_number, best_cos) = best
        if points_number > best_points_number:
            best = res
    return best


def find_best_views(corner_storage, intrinsic_mat):
    print("Searching best view for initialization")
    frames_number = len(corner_storage)
    results = []
    for f1, f2 in generate_frame_pairs_with_fixed_init(frames_number):
        print(f"Checking {f1, f2} / {frames_number}")
        success, pose1, pose2 = get_views_by_frames(corner_storage, intrinsic_mat, f1, f2)
        if not success:
            continue
        result = estimate_views(corner_storage, intrinsic_mat, pose1, pose2, f1, f2)
        results.append(((f1, pose1), (f2, pose2), result))
    return get_best_result(results)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    print(f"Building camera track and point of clouds for {len(corner_storage)} frames")
    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2, result = find_best_views(corner_storage, intrinsic_mat)
        if known_view_1 is None or known_view_2 is None:
            raise ValueError("Could not find any frames to initialize")
        print(f"Initial number of points value is {result}. Initialized with frames {known_view_1[0]}, {known_view_2[0]}")
    else:
        print(f"Intializing with given views: {known_view_1[0]}, {known_view_2[0]}")

    frames_number = len(rgb_sequence)
    inliers, view_mats, point_cloud_builder = [None] * frames_number, [None] * frames_number, PointCloudBuilder(np.array([]).astype(np.int64))

    frame1, pos1 = known_view_1
    frame2, pos2 = known_view_2
    view_mats[frame1] = pose_to_view_mat3x4(pos1)
    inliers[frame1] = 0
    view_mats[frame2] = pose_to_view_mat3x4(pos2)
    inliers[frame2] = 0

    try_add_points(point_cloud_builder, corner_storage, view_mats, intrinsic_mat, frame1, frame2, params)
    print(f"PC size after first step is: {point_cloud_builder.points.shape[0]}")

    updated = True
    epoch = 0
    LIMIT = 2
    while updated and epoch < LIMIT:
        epoch += 1
        updated = False
        print(f"Current epoch: {epoch}")
        for frame in range(frames_number):
            initial_size = point_cloud_builder.points.shape[0]
            status = try_set_view_matrix(point_cloud_builder, corner_storage, view_mats, inliers, intrinsic_mat, frame)
            if status:
                updated = True
            for frame2 in range(frames_number):
                old_size = point_cloud_builder.points.shape[0]
                try_add_points(point_cloud_builder, corner_storage, view_mats, intrinsic_mat, frame, frame2, params)
                new_size = point_cloud_builder.points.shape[0]
                if old_size != new_size:
                    updated = True
            print(f'Processed {frame + 1} frames of {frames_number} in epoch {epoch}: PC size {initial_size} -> {point_cloud_builder.points.shape[0]}')

        views_unset = len(list(filter(lambda mat: mat is None, view_mats)))
        print(f'After epoch {epoch} {frames_number - views_unset} are set')

    unset_views_number = len(list(filter(lambda mat: mat is None, view_mats)))
    if unset_views_number != 0:
        raise ValueError(f"Failed to find view matrix for {unset_views_number} frames")

    print(f"Building done. All view matrix are set, PC size is {point_cloud_builder.points.shape[0]}")
    view_mats = run_bundle_adjustment(intrinsic_mat, corner_storage, view_mats, point_cloud_builder)

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
