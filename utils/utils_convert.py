import sys
import math
import torch
import argparse
import numpy as np
import torchvision
from bvh import Bvh
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as R


JOINT_MAP = {
    # 'BVH joint name': 'SMPLX joint index'
    'Hips': 0,
    'LeftUpLeg': 1,
    'RightUpLeg': 2,
    'Spine': 3,
    'LeftLeg': 4,
    'RightLeg': 5,
    'Spine1': 6,
    'LeftFoot': 7,
    'RightFoot': 8,
    'Spine2': 9,
    'LeftToeBase': 10,
    'RightToeBase': 11,
    'Neck': 12,
    'LeftShoulder': 13,
    'RightShoulder': 14,
    'Head': 15,
    'LeftArm': 16,
    'RightArm': 17,
    'LeftForeArm': 18,
    'RightForeArm': 19,
    'LeftHand': 20,
    'RightHand': 21,
}

JOINT_MAP_UPEER_FIRST = {
    "Pelvis": 0,
    "LeftHip": 1,
    "RightHip": 2,
    "Spine": 3,
    "LeftLeg": 4,
    "RightLeg": 5,
    "Spine1": 6,
    "LeftFoot": 7,
    "RightFoot": 8,
    "Spine2": 9,
    "LeftToeBase": 10,
    "RightToeBase": 11,
    "Neck": 12,
    "LeftShoulder": 13,
    "RightShoulder": 14,
    "Head": 15,
    "LeftArm": 16,
    "RightArm": 17,
    "LeftForeArm": 18,
    "RightForeArm": 19,
    "LeftHand": 20,
    "RightHand": 21,
}

def bvh_to_smplx(bvh_file):
    with open(bvh_file, 'r') as f:
        mocap = Bvh(f.read())
    num_frames = len(mocap.frames)
    print('Number of bvh frames:', num_frames)
    print('Only use the first 300 frames.')
    num_frames = min(num_frames, 300)
    bvh_joint_names = set(mocap.get_joints_names())
    smplx_poses = np.zeros((num_frames, len(JOINT_MAP.keys()), 3))
    for i in range(0, num_frames):
        for joint_name, joint_index in JOINT_MAP.items():
            if joint_name not in bvh_joint_names:
                if i == 0:
                    print(joint_name)
                continue
            rotation = transforms.euler_angles_to_matrix(
                torch.tensor(mocap.frame_joint_channels(
                    i, joint_name, ['Zrotation', 'Xrotation', 'Yrotation']
                ))/180*math.pi, 'ZXY'
            )
            rotation = transforms.matrix_to_axis_angle(rotation).numpy()
            smplx_poses[i, joint_index] = rotation

    for i in range(0, num_frames):
        for joint_name, joint_index in JOINT_MAP_UPEER_FIRST.items():
            if joint_name not in bvh_joint_names:
                if i == 0:
                    print(joint_name)
                continue
            rotation = transforms.euler_angles_to_matrix(
                torch.tensor(mocap.frame_joint_channels(
                    i, joint_name, ['Zrotation', 'Xrotation', 'Yrotation']
                ))/180*math.pi, 'ZXY'
            )
            rotation = transforms.matrix_to_axis_angle(rotation).numpy()
            smplx_poses[i, joint_index] = rotation
    # get world positions
    # bvh_joints_params = get_joint_positions(mocap)
    bvh_joints_params = None


    return smplx_poses.astype(np.float32), bvh_joints_params


def get_joint_positions(bvh_tree, scale=1.0, end_sites=False):
    time_col = np.arange(0, (bvh_tree.nframes - 0.5) * bvh_tree.frame_time, bvh_tree.frame_time)[:, None]
    data_list = [time_col]
    header = ['time']
    root = next(bvh_tree.root.filter('ROOT'))
    
    def get_world_positions(joint):
        if joint.value[0] == 'End':
            joint.world_transforms = np.tile(t3d.affines.compose(np.zeros(3), np.eye(3), np.ones(3)),
                                             (bvh_tree.nframes, 1, 1))
        else:
            channels = bvh_tree.joint_channels(joint.name)
            axes_order = ''.join([ch[:1] for ch in channels if ch[1:] == 'rotation']).lower()  # FixMe: This isn't going to work when not all rotation channels are present
            axes_order = 's' + axes_order[::-1]
            joint.world_transforms = get_affines(bvh_tree, joint.name, axes=axes_order)
            
        if joint != root:
            # For joints substitute position for offsets.
            offset = [float(o) for o in joint['OFFSET']]
            joint.world_transforms[:, :3, 3] = offset
            joint.world_transforms = np.matmul(joint.parent.world_transforms, joint.world_transforms)
        if scale != 1.0:
            joint.world_transforms[:, :3, 3] *= scale
            
        header.extend(['{}.{}'.format(joint.name, channel) for channel in 'xyz'])
        pos = joint.world_transforms[:, :3, 3]
        data_list.append(pos)
        
        if end_sites:
            end = list(joint.filter('End'))
            if end:
                get_world_positions(end[0])  # There can be only one End Site per joint.
        for child in joint.filter('JOINT'):
            get_world_positions(child)
    
    get_world_positions(root)
    data = np.concatenate(data_list, axis=1)
    return data



def get_rotation_matrices(bvh_tree, joint_name, axes='rzxz'):
    """Read the Euler angles of a joint in order given by axes and return it as rotation matrices for all frames.

    :param bvh_tree: BVH structure.
    :type bvh_tree: bvhtree.BvhTree
    :param joint_name: Name of the joint.
    :type joint_name: str
    :param axes: The order in which to return the angles. Usually that's the joint's channel order.
    :type axes: str
    :return: rotation matrix (frames x 3 x 3)
    :rtype: numpy.ndarray
    """
    import transforms3d as t3d
    from itertools import repeat
    def prune(a, epsilon=0.00000001):
        """Sets absolute values smaller than epsilon to 0.
        It does this in-place on the input array.

        :param a: array
        :type a: numpy.ndarray
        :param epsilon: threshold
        :type epsilon: float
        """
        a[np.abs(a) < epsilon] = 0.0
    eulers = np.radians(get_euler_angles(bvh_tree, joint_name, axes[1:]))
    matrices = np.array(list(map(t3d.euler.euler2mat, eulers[:, 0], eulers[:, 1], eulers[:, 2], repeat(axes))))
    prune(matrices)
    return matrices


def get_translations(bvh_tree, joint_name):
    """Get the xyz translation of a joint for all frames.
    
    :param bvh_tree: BVH structure.
    :type bvh_tree: bvhtree.BvhTree
    :param joint_name: Name of the joint.
    :type joint_name: str
    :return: translations xyz for all frames (frames x 3).
    :rtype: numpy.ndarray
    """
    translations = np.array(bvh_tree.frame_joint_channels(0, joint_name, ['Xposition', 'Yposition', 'Zposition'],
                                                           ))  # For missing channels. bvh > v3.0!
    return translations
    
    
def get_affines(bvh_tree, joint_name, axes='rzxz'):
    """Read the transforms of a joint with rotation in order given by axes and return it as an affine matrix.

    :param bvh_tree: BVH structure.
    :type bvh_tree: bvhtree.BvhTree
    :param joint_name: Name of the joint.
    :type joint_name: str
    :param axes: The order in which to return the angles. Usually that's the joint's channel order.
    :type axes: str
    :return: affine matrix (frames x 4 x 4)
    :rtype: numpy.ndarray
    """
    import transforms3d as t3d
    translations = get_translations(bvh_tree, joint_name)
    rot_matrices = get_rotation_matrices(bvh_tree, joint_name, axes=axes)
    
    affine_matrices = np.array(list(map(t3d.affines.compose,
                                        translations,
                                        rot_matrices,
                                        np.ones((bvh_tree.nframes, 3)))))
    return affine_matrices


def reorder_axes(xyz, axes='zxy'):
    """Takes an input array in xyz order and re-arranges it to given axes order.

    :param xyz: array with x,y,z order.
    :param axes: output order for Euler conversions respective to axes.
    :return: array in axes order.
    :rtype: numpy.ndarray
    """
    if xyz.shape[-1] != 3 or len(xyz.shape) > 2:
        print("ERROR: Frames must be 1D or 2D array with 3 columns for x,y,z axes!")
        raise ValueError
    # If the output order is the same as the input, do not reorder.
    if axes == 'xyz':
        return xyz
    
    i, j, k = _get_reordered_indices(axes)
    # 1-D arrays.
    if len(xyz.shape) == 1:
        res = np.array([xyz[i], xyz[j], xyz[k]])
    # 2-D arrays: multiple frames.
    elif len(xyz.shape) == 2:
        # Todo: is something like data[:,1], data[:,2] = data[:,2], data[:,1].copy() faster/more efficient?
        res = np.array([xyz[:, i], xyz[:, j], xyz[:, k]]).T
    return res


def _get_reordered_indices(rotation_order):
    """Returns indices for converting 'xyz' rotation order to given rotation order.
    
    :param rotation_order: Rotation order to convert to.
    :type rotation_order: str
    :return: Indices for getting from xyz to given axes rotation order.
    :rtype: tuple
    """
    # Axis sequences for Euler angles.
    _NEXT_AXIS = [1, 2, 0, 1]

    # Map axes strings to/from tuples of inner axis, parity.
    _AXES2TUPLE = {
        'xyz': (0, 0), 'xyx': (0, 0), 'xzy': (0, 1),
        'xzx': (0, 1), 'yzx': (1, 0), 'yzy': (1, 0),
        'yxz': (1, 1), 'yxy': (1, 1), 'zxy': (2, 0),
        'zxz': (2, 0), 'zyx': (2, 1), 'zyz': (2, 1)}
    try:
        firstaxis, parity = _AXES2TUPLE[rotation_order]
    except KeyError:
        print("Rotation order must be one of {}.".format(', '.join(_AXES2TUPLE.keys())))
        raise
    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]
    return i, j, k


def get_euler_angles(bvh_tree, joint_name, axes='zxy'):
    """Return Euler angles in degrees for joint in all frames.

    :param bvh_tree: BVH structure.
    :type bvh_tree: bvhtree.BvhTree
    :param joint_name: Name of the joint
    :type joint_name: str
    :param axes: The order in which to return the angles. Usually that's the joint's channel order.
    :return: Euler angles in order of axes (frames x 3).
    :rtype: numpy.ndarray
    """
    euler_xyz = np.array(bvh_tree.frame_joint_channels(joint_name, ['Xrotation', 'Yrotation', 'Zrotation'],
                                                        ))  # For missing channels. bvh > v3.0!
    euler_ordered = reorder_axes(euler_xyz, axes)
    return euler_ordered

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_path", '-b', default=None, type=str)
    args = parser.parse_args()
    smplx_poses = bvh_to_smplx(args.bvh_path)
    np.save(args.bvh_path.replace('.bvh', '.npy'), smplx_poses.astype(np.float32))
