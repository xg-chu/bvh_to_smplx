import os
import torch
import random
import argparse
import numpy as np
import torchvision
from tqdm import tqdm
from smplx import SMPLX
from utils.utils_convert import bvh_to_smplx
from utils.utils_vis import Mesh_Renderer, plot_3d_motion
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform

class SMPLX_Renderer(torch.nn.Module):
    def __init__(self, asset_path, gender='neutral', device='cuda'):
        super().__init__()
        self.device = device
        self.smplx = SMPLX(
            asset_path, gender=gender, use_face_contour=False,
            num_betas=10, num_expression_coeffs=10, ext='npz', use_pca=False
        ).to(device)
        self.renderer = Mesh_Renderer(image_size=1024, faces=self.get_faces(), device=device)

    def get_faces(self, ):
        return self.smplx.faces.astype(np.int32)
    
    def forward(
            self, pose_param=None, camera=None, global_orient=None, 
            left_hand_pose=None, right_hand_pose=None, 
            render_mesh=False
        ):
        if camera is None:
            camera = build_camera(device=self.device)
        shape_param = pose_param.new_zeros(1, 10)
        if global_orient is None:
            global_orient = pose_param.new_zeros(1, 1, 3)
        else:
            global_orient = global_orient
        transl = pose_param.new_zeros(pose_param.shape[0], 3)
        transl[..., 1] += 0.4
        output = self.smplx(
            betas=shape_param,
            expression=None, jaw_pose=None,
            global_orient=global_orient, body_pose=pose_param, transl=transl,
            right_hand_pose=right_hand_pose, left_hand_pose=left_hand_pose,
            leye_pose=shape_param.new_zeros(shape_param.shape[0], 3), reye_pose=shape_param.new_zeros(shape_param.shape[0], 3),
            return_verts=True
        )
        vertices = output.vertices.detach()
        joints = output.joints.detach()
        joints = joints * torch.tensor([-1, 1, 1], device=joints.device).view(1, 1, 3)
        if render_mesh:
            render_res, _ = self.renderer(vertices, camera)
        else:
            render_res = None
        return vertices, joints, render_res


def build_camera(device='cuda'):
    R, T = look_at_view_transform(1.5, 30, 45) # d, e, a
    camera = PerspectiveCameras(device=device, R=R, T=T)
    return camera


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_path", '-p', default=None, type=str)
    parser.add_argument("--render_mesh", action='store_true')
    parser.add_argument("--render_fps", default=20, type=int)
    args = parser.parse_args()

    smplx_renderer = SMPLX_Renderer('./assets')
    # convert bvh to smplx parameters
    print('Converting bvh to smplx parameters ...')
    pose_params, bvh_joints_params = bvh_to_smplx(args.pose_path)
    pose_params = torch.tensor(pose_params).cuda()
    # build smplx
    print('Building SMPLX vertices and parameters...')
    all_vertices, all_joints, images = [], [], []
    for pose_param in tqdm(pose_params):
        vertices, joints, render_image = smplx_renderer(
            pose_param[1:22][None], # global_orient=pose_param[:1][None],
            render_mesh=args.render_mesh
        )
        images.append(render_image)
        all_joints.append(joints)
        all_vertices.append(vertices)
    all_joints = torch.cat(all_joints, dim=0)[:, :22]
    all_vertices = torch.cat(all_vertices, dim=0)
    all_vertices = all_vertices[:, random.sample(range(all_vertices.shape[1]), 5000)]
    if args.render_mesh:
        all_images = torch.cat(images, dim=0).permute(0, 2, 3, 1).to(torch.uint8).cpu()
    else:
        all_images = plot_3d_motion(None, all_joints)
        # all_images = plot_3d_motion(all_vertices, None)
        all_images = all_images.permute(0, 2, 3, 1).to(torch.uint8)
    base_name = os.path.basename(args.pose_path).split('.')[0]
    torchvision.io.write_video(f'./{base_name}.mp4', all_images, fps=args.render_fps)
    print(f'Results saved to {base_name}.mp4')
