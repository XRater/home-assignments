from typing import List, Dict

from collections import namedtuple
import numpy as np
from scipy.optimize import approx_fprime as derivative
import sortednp as snp

from _camtrack import (
    calc_inlier_indices,
    PointCloudBuilder,
    rodrigues_and_translation_to_view_mat3x4,
    view_mat3x4_to_rodrigues_and_translation,
    compute_reprojection_errors,
)

ProjectionError = namedtuple(
    'ProjectionError',
    ('frame_id', 'id_3d', 'id_2d')
)

MAX_PROJECTION_ERROR = 1.0
STEPS = 4
LAMBDA = 10


def collect_projection_errors(point_cloud_builder, corner_storage, view_mats, intrinsic_mat):
    frames_number = len(view_mats)
    points = point_cloud_builder.points
    points_ids = point_cloud_builder.ids
    errors = []
    for frame in range(frames_number):
        view_mat = view_mats[frame]
        frame_corners = corner_storage[frame]

        _, (indices_1, indices_2) = snp.intersect(points_ids.flatten(), frame_corners.ids.flatten(), indices=True)
        points_3d = points[indices_1]
        points_2d = frame_corners.points[indices_2]

        inliers = calc_inlier_indices(points_3d, points_2d, intrinsic_mat @ view_mat, MAX_PROJECTION_ERROR)
        for inlier in inliers:
            errors.append(ProjectionError(frame, indices_1[inlier], indices_2[inlier]))

    used = set()
    for error in errors:
        used.add(error.id_3d)
    used = list(sorted(used))

    return errors, used


def run_bundle_adjustment(intrinsic_mat, corner_storage, view_mats, point_cloud_builder):
    errors, used = collect_projection_errors(point_cloud_builder, corner_storage, view_mats, intrinsic_mat)
    view_mats_vector = np.array([view_mat_to_vector(view_mat) for view_mat in view_mats])
    points3d_vector = np.array(point_cloud_builder.points[used])
    indices = np.zeros(point_cloud_builder.ids.max()).astype(np.int32) - 1
    for k, point_index in enumerate(used):
        indices[point_index] = k

    view_mats_vector, points3d_vector = \
        optimize(errors, corner_storage, intrinsic_mat, view_mats_vector, points3d_vector, indices)

    view_mats = [vector_to_view_mat(vector) for vector in view_mats_vector]
    point_cloud_builder.update_points(point_cloud_builder.ids[used], points3d_vector)

    return view_mats


def view_mat_to_vector(view_mat):
    r, t = view_mat3x4_to_rodrigues_and_translation(view_mat)
    return np.concatenate([r.squeeze(), t])


def vector_to_view_mat(vector):
    r, t = vector[0:3].reshape(3, 1), vector[3:6].reshape(3, 1)
    return rodrigues_and_translation_to_view_mat3x4(r, t)


def args_to_vector(view_mat, point3d):
    view_mat_vec = view_mat_to_vector(view_mat)
    return np.concatenate([view_mat_vec, point3d])


def vector_to_args(vector):
    r, t, point = vector[0:3].reshape(3, 1), vector[3:6].reshape(3, 1), vector[6:9]
    return rodrigues_and_translation_to_view_mat3x4(r, t), point


def get_error_from_vec(vector, point2d, intrinsic_mat) -> np.float32:
    view_mat, point3d = vector_to_args(vector)
    return compute_reprojection_errors(point3d.reshape(1, -1), point2d.reshape(1, -1), intrinsic_mat @ view_mat)[0]


def get_errors(projection_errors, corner_storage, intrinsic_mat, view_mats_vector, points_vector, indices):
    errors = []
    for proj_err in projection_errors:
        view_mat = vector_to_view_mat(view_mats_vector[proj_err.frame_id])
        point3d = points_vector[indices[proj_err.id_3d]]
        point2d = corner_storage[proj_err.frame_id].points[proj_err.id_2d]
        vector = args_to_vector(view_mat, point3d)
        errors.append(get_error_from_vec(vector, point2d, intrinsic_mat))
    return np.array(errors)


def jacobian(projection_errors, corner_storage, intrinsic_mat, view_mats_vector, points_vector, indices):
    view_mats_args = len(view_mats_vector.reshape(-1))
    points_args = len(points_vector.reshape(-1))
    J = np.zeros((len(projection_errors), view_mats_args + points_args))

    for row, proj_err in enumerate(projection_errors):
        view_mat = vector_to_view_mat(view_mats_vector[proj_err.frame_id])
        point3d = points_vector[indices[proj_err.id_3d]]
        point2d = corner_storage[proj_err.frame_id].points[proj_err.id_2d]
        vector = args_to_vector(view_mat, point3d)
        loss_function = lambda v: get_error_from_vec(v, point2d, intrinsic_mat)
        derivatives = derivative(vector, loss_function, np.full_like(vector, 1e-9))

        view_mat_pos = 6 * proj_err.frame_id
        vector_pos = view_mats_args + 3 * indices[proj_err.id_3d]
        J[row, view_mat_pos:view_mat_pos + 6] = derivatives[:6]
        J[row, vector_pos:vector_pos + 3] = derivatives[6:]

    return J


def update_vector(vector, dx):
    shape = vector.shape
    return (vector.reshape(-1) + dx).reshape(shape)


def get_vectors_updates(J, errors, size):
    try:
        JTJ = J.T @ J
        JTJ += LAMBDA * np.diag(np.diag(JTJ))
        U, W, V = JTJ[:size, :size], JTJ[:size, size:], JTJ[size:, size:]
        Vi = np.zeros_like(V)
        for i in range(0, len(V), 3):
            s = 3 * i
            Vi[s:s + 3, s:s + 3] = np.linalg.inv(V[s:s + 3, s:s + 3])
        g = J.T @ errors
        A = U - W @ Vi @ W.T
        b = W @ Vi @ g[size:] - g[:size]
        dc = np.linalg.solve(A, b)
        dx = Vi @ (-g[size:] - W.T @ dc)
    except np.linalg.LinAlgError:
        return None, None
    return dc, dx


def optimize(projection_errors, corner_storage, intrinsic_mat, view_mats_vector, points_vector, indices):
    size = len(view_mats_vector.reshape(-1))
    errors = get_errors(projection_errors, corner_storage, intrinsic_mat, view_mats_vector, points_vector, indices)
    print(f'Mean error before adjustment is {errors.mean()}')
    for step in range(STEPS):
        J = jacobian(projection_errors, corner_storage, intrinsic_mat, view_mats_vector, points_vector, indices)
        errors = get_errors(projection_errors, corner_storage, intrinsic_mat, view_mats_vector, points_vector, indices)
        dc, dx = get_vectors_updates(J, errors, size)
        if dc is None or dx is None:
            continue
        view_mats_vector = update_vector(view_mats_vector, dc)
        points_vector = update_vector(points_vector, dx)

    errors = get_errors(projection_errors, corner_storage, intrinsic_mat, view_mats_vector, points_vector, indices)
    print(f'Mean error after adjustment is {errors.mean()}')

    return view_mats_vector, points_vector

