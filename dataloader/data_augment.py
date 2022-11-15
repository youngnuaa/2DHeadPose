from __future__ import division
import cv2
import numpy as np
import scipy
from scipy import ndimage
import math



def random_crop(x, dn):
    dx = np.random.randint(dn, size=1)[0]
    dy = np.random.randint(dn, size=1)[0]
    h = x.shape[0]
    w = x.shape[1]
    out = x[0 + dy:h - (dn - dy), 0 + dx:w - (dn - dx), :]
    out = cv2.resize(out, (h, w), interpolation=cv2.INTER_CUBIC)
    return out


def random_crop_black(x, dn):
    dx = np.random.randint(dn, size=1)[0]
    dy = np.random.randint(dn, size=1)[0]

    h = x.shape[0]
    w = x.shape[1]

    dx_shift = np.random.randint(dn, size=1)[0]
    dy_shift = np.random.randint(dn, size=1)[0]
    out = x * 0
    out[0 + dy_shift:h - (dn - dy_shift), 0 + dx_shift:w - (dn - dx_shift), :] = x[0 + dy:h - (dn - dy),
                                                                                 0 + dx:w - (dn - dx), :]

    return out


def random_crop_white(x, dn):
    dx = np.random.randint(dn, size=1)[0]
    dy = np.random.randint(dn, size=1)[0]
    h = x.shape[0]
    w = x.shape[1]

    dx_shift = np.random.randint(dn, size=1)[0]
    dy_shift = np.random.randint(dn, size=1)[0]
    out = x * 0 + 255
    out[0 + dy_shift:h - (dn - dy_shift), 0 + dx_shift:w - (dn - dx_shift), :] = x[0 + dy:h - (dn - dy),
                                                                                 0 + dx:w - (dn - dx), :]

    return out


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 - 0.5
    o_y = float(y) / 2 - 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                           row_axis=1, col_axis=2, channel_axis=0,
                           fill_mode='nearest', cval=0., order=1):
    """Applies an affine transformation specified by the parameters given.
    # Arguments
        x: 3D numpy array - a 2D image with one or more channels.
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        row_axis: Index of axis for rows (aka Y axis) in the input image.
                  Direction: left to right.
        col_axis: Index of axis for columns (aka X axis) in the input image.
                  Direction: top to bottom.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        order: int, order of interpolation
    # Returns
        The transformed version of the input.
    """
    if scipy is None:
        raise ImportError('Image transformations require SciPy. '
                          'Install SciPy.')

    # Input sanity checks:
    # 1. x must 2D image with one or more channels (i.e., a 3D tensor)
    # 2. channels must be either first or last dimension
    if np.unique([row_axis, col_axis, channel_axis]).size != 3:
        raise ValueError("'row_axis', 'col_axis', and 'channel_axis'"
                         " must be distinct")

    # TODO: shall we support negative indices?
    valid_indices = set([0, 1, 2])
    actual_indices = set([row_axis, col_axis, channel_axis])
    if actual_indices != valid_indices:
        raise ValueError(f"Invalid axis' indices: {actual_indices - valid_indices}")

    if x.ndim != 3:
        raise ValueError("Input arrays must be multi-channel 2D images.")
    if channel_axis not in [0, 2]:
        raise ValueError("Channels are allowed and the first and last dimensions.")

    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x = np.rollaxis(x, channel_axis, 0)

        # Matrix construction assumes that coordinates are x, y (in that order).
        # However, regular numpy arrays use y,x (aka i,j) indexing.
        # Possible solution is:
        #   1. Swap the x and y axes.
        #   2. Apply transform.
        #   3. Swap the x and y axes again to restore image-like data ordering.
        # Mathematically, it is equivalent to the following transformation:
        # M' = PMP, where P is the permutation matrix, M is the original
        # transformation matrix.
        if col_axis > row_axis:
            transform_matrix[:, [0, 1]] = transform_matrix[:, [1, 0]]
            transform_matrix[[0, 1]] = transform_matrix[[1, 0]]
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='nearest', cval=0., interpolation_order=1):
    """Performs a random spatial zoom of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interpolation_order: int, order of spline interpolation.
            see `ndimage.interpolation.affine_transform`
    # Returns
        Zoomed Numpy image tensor.
    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two'
                         ' floats. Received: %s' % (zoom_range,))

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    x = apply_affine_transform(x,
                               zx=zx,
                               zy=zy,
                               row_axis=row_axis,
                               col_axis=col_axis,
                               channel_axis=channel_axis,
                               fill_mode=fill_mode,
                               cval=cval,
                               order=interpolation_order)
    return x


def augment_data(image):
    rand_r = np.random.random()
    if rand_r < 0.25:
        dn = np.random.randint(52, size=1)[0] + 1
        image = random_crop(image, dn)

    elif rand_r >= 0.25 and rand_r < 0.5:
        dn = np.random.randint(52, size=1)[0] + 1
        image = random_crop_black(image, dn)

    elif rand_r >= 0.5 and rand_r < 0.75:
        dn = np.random.randint(52, size=1)[0] + 1
        image = random_crop_white(image, dn)

    if np.random.random() > 0.3:
        image = random_zoom(image, [0.8, 1.2], row_axis=0, col_axis=1, channel_axis=2)

    return image


def gen_cos_sin_value(angle):
    cos_v = np.cos(angle)
    sin_v = np.sin(angle)
    return cos_v, sin_v


def augment_angle_data(yaw, pitch, roll, angle):
    yaw = yaw / 180 * np.pi
    pitch = pitch / 180 * np.pi
    roll = roll / 180 * np.pi
    angle = angle / 180 * np.pi

    cos_roll, sin_roll = gen_cos_sin_value(roll)
    cos_pitch, sin_pitch = gen_cos_sin_value(pitch)
    cos_yaw, sin_yaw = gen_cos_sin_value(yaw)
    cos_angle, sin_angle = gen_cos_sin_value(angle)

    y_matrix = [[cos_yaw, 0, -sin_yaw],
              [0, 1, 0],
              [sin_yaw, 0, cos_yaw] ]

    p_matrix = [[1, 0, 0],
              [0, cos_pitch, sin_pitch],
              [0, -sin_pitch, cos_pitch] ]

    r_matrix = [[cos_roll, sin_roll, 0],
              [-sin_roll, cos_roll, 0],
              [0, 0, 1] ]

    an_matrix = [[cos_angle, sin_angle, 0],
              [-sin_angle, cos_angle, 0],
              [0, 0, 1]]

    t = np.dot(r_matrix, y_matrix)
    t = np.dot(t, p_matrix)
    t = np.dot(t, an_matrix)

    pitch = -math.atan2(t[2, 1], t[2, 2]) * 180 / np.pi
    yaw = math.asin(t[2, 0]) * 180 / np.pi
    roll = -math.atan2(t[1, 0], t[0, 0]) * 180 / np.pi


    return [yaw, pitch, roll]



