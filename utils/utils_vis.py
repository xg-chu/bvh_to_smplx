# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
import os
import torch
import torch.nn as nn
from pytorch3d.io import load_obj
from pytorch3d.structures import (
    Meshes, Pointclouds
)
from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform, RasterizationSettings, 
    FoVPerspectiveCameras, PointsRasterizationSettings,
    PointsRenderer, PointsRasterizer, AlphaCompositor,
    PointLights, AmbientLights, TexturesVertex, TexturesUV, 
    SoftPhongShader, MeshRasterizer, MeshRenderer, SoftSilhouetteShader
)

class Mesh_Renderer(nn.Module):
    def __init__(self, image_size, obj_filename=None, faces=None, device='cpu'):
        super(Mesh_Renderer, self).__init__()
        self.device = device
        self.image_size = image_size
        if obj_filename is not None:
            verts, faces, aux = load_obj(obj_filename, load_textures=False)
            self.faces = faces.verts_idx
        elif faces is not None:
            if isinstance(faces, torch.Tensor):
                self.faces = faces.long()
            else:
                import numpy as np
                self.faces = torch.tensor(faces.astype(np.int32))
        else:
            raise NotImplementedError('Must have faces.')
        self.raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
        self.lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

    def _build_cameras(self, transform_matrix, focal_length, principal_point):
        batch_size = transform_matrix.shape[0]
        screen_size = torch.tensor(
            [self.image_size, self.image_size], device=self.device
        ).float()[None].repeat(batch_size, 1)
        if type(principal_point) == torch.Tensor:
            principal_point = principal_point.to(self.device).float()
        else:
            principal_point = torch.tensor([principal_point], device=self.device)
        cameras_kwargs = {
            'principal_point': principal_point.repeat(batch_size, 1), 'focal_length': focal_length, 
            'image_size': screen_size, 'device': self.device,
        }
        cameras = PerspectiveCameras(**cameras_kwargs, R=transform_matrix[:, :3, :3], T=transform_matrix[:, :3, 3])
        return cameras

    def forward(
            self, vertices, cameras=None, 
            transform_matrix=None, focal_length=None, principal_point=None
        ):
        if cameras is None:
            cameras = self._build_cameras(transform_matrix, focal_length, principal_point)
        faces = self.faces[None].repeat(vertices.shape[0], 1, 1)
        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(vertices)  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
        mesh = Meshes(
            verts=vertices.to(self.device),
            faces=faces.to(self.device),
            textures=textures
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings),
            shader=SoftPhongShader(cameras=cameras, lights=self.lights, device=self.device)
        )
        render_results = renderer(mesh).permute(0, 3, 1, 2)
        images = render_results[:, :3]
        alpha_images = render_results[:, 3:]
        # images = images*alpha_images
        return images*255, alpha_images


class Point_Renderer(nn.Module):
    def __init__(self, image_size=256, device='cpu'):
        super(Point_Renderer, self).__init__()
        self.device = device
        R, T = look_at_view_transform(4, 30, 30) # d, e, a
        self.cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01, zfar=1.0)
        raster_settings = PointsRasterizationSettings(
            image_size=image_size, radius=0.005, points_per_pixel=10
        )
        rasterizer = PointsRasterizer(cameras=self.cameras, raster_settings=raster_settings)
        self.renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
        
    def forward(self, points, D=3, E=15, A=30, coords=True, ex_points=None):
        if D !=8 or E != 30 or A != 30:
            R, T = look_at_view_transform(D, E, A) # d, e, a
            self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, znear=0.01, zfar=1.0)
        verts = torch.Tensor(points).to(self.device)
        verts = verts[:, torch.randperm(verts.shape[1])[:10000]]
        if ex_points is not None:
            verts = torch.cat([verts, ex_points.expand(verts.shape[0], -1, -1)], dim=1)
        if coords:
            coords_size = verts.shape[1]//10
            cod = verts.new_zeros(coords_size*3, 3)
            li = torch.linspace(0, 1.0, steps=coords_size, device=cod.device)
            cod[:coords_size, 0], cod[coords_size:coords_size*2, 1], cod[coords_size*2:, 2] = li, li, li
            verts = torch.cat(
                [verts, cod.unsqueeze(0).expand(verts.shape[0], -1, -1)], dim=1
            )
        rgb = torch.Tensor(torch.rand_like(verts)).to(self.device)
        point_cloud = Pointclouds(points=verts, features=rgb)
        images = self.renderer(point_cloud, cameras=self.cameras).permute(0, 3, 1, 2)
        return images*255


def plot_3d_motion(vertices=None, joints=None, figsize=(4, 4), radius=4):
    import io
    import random
    import torchvision
    import numpy as np
    from PIL import Image
    import matplotlib
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    random.seed(40)
    # colors = ['red', 'blue', 'black', 'red', 'blue', 'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkred', 'darkred','darkred','darkred','darkred']
    colors = random.sample(matplotlib.colors.cnames.keys(), 22)
    kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

    # pytorch3d xyz to matplotlib xyz
    if vertices is not None:
        vertices = vertices.cpu().numpy().copy()
        vertices = vertices[..., [0, 2, 1]]
        vertices[..., [0, 1]] *= -1
        frame_number = vertices.shape[0]
    if joints is not None:
        joints = joints.cpu().numpy().copy()
        joints = joints[..., [0, 2, 1]]; 
        joints[..., [0, 1]] *= -1
        frame_number = joints.shape[0]
    # visualize
    all_images = []
    for index in tqdm(range(frame_number)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        eps = 1e-16
        ax.axes.set_xlim3d(left=-1.-eps, right=1.+eps)
        ax.axes.set_ylim3d(bottom=-1.-eps, top=1.+eps) 
        ax.axes.set_zlim3d(bottom=-1.-eps, top=1.+eps) 
        # ax.view_init(elev=None, azim=135)
        n = 100
        if joints is not None:
            # for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            #     linewidth = 4.0 if i < 5 else 2.0
            #     ax.plot3D(joints[index, chain, 0], joints[index, chain, 1], joints[index, chain, 2], linewidth=linewidth, color=color)
            for i, chain in enumerate(kinematic_tree):
                linewidth = 4.0 if i < 5 else 2.0
                for cid in range(len(chain)-1):
                    chains = chain[cid:cid+2]
                    color = colors[chain[cid+1]]
                    ax.plot3D(joints[index, chains, 0], joints[index, chains, 1], joints[index, chains, 2], linewidth=linewidth, color=color)
        if vertices is not None:
            ax.scatter(vertices[index, :, 0], vertices[index, :, 1], vertices[index, :, 2], marker='o', alpha=0.05)
        
        # plt.axis('off')    
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200)
        buf.seek(0)
        plt.close()
        image = torchvision.transforms.functional.pil_to_tensor(Image.open(buf))
        all_images.append(image[:3])
    return torch.stack(all_images)
    