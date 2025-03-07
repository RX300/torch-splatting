import os
import json
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from einops import rearrange

# change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
def read_camera(folder):
    """
    read camera from json file
    """
    scene_info = json.load(open(os.path.join(folder, 'info.json')))
    max_depth = 1
    try:
        max_depth = scene_info['images'][0]['max_depth']
    except:
        pass

    rgb_files = []
    poses = []
    intrinsics = []
    for item in scene_info['images']:
        rgb_files.append(os.path.join(folder, item['rgb']))
        c2w = item['pose']
        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)
        poses.append(np.array(c2w_opencv))
        intrinsics.append(np.array(item['intrinsic']))
    return rgb_files, poses, intrinsics, max_depth

# change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
def read_camera_tinynerf(folder,max_depth=1):
    """
    read camera from json file
    """
    data = np.load(folder)
    data_images = data['images']
    data_poses = data['poses']
    size = data_images.shape[0]
    H,W=data_images.shape[1],data_images.shape[2]
    focal = float(data['focal'])
    rgb_files = []
    poses = []
    intrinsics = []
    for i in range(size):
        rgb_files.append(data_images[i])
        c2w = data_poses[i]
        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)
        poses.append(np.array(c2w_opencv))
        intrinsic = np.eye(3)
        intrinsic[0, 0] = focal  # fx
        intrinsic[1, 1] = focal  # fy
        intrinsic[0, 2] = W / 2  # cx
        intrinsic[1, 2] = H / 2  # cy
        intrinsics.append(np.array(intrinsic))

    return rgb_files, poses, intrinsics, max_depth

def read_all(folder, resize_factor=1.):
    """
    read source images from a folder
    """
    # scene_src_dir = os.path.join(self.folder_path_src, scene_id)
    src_rgb_files, src_poses, src_intrinsics, max_depth = read_camera(folder)

    src_cameras = []
    src_rgbs = []
    src_alphas = []
    src_depths = []
    for src_rgb_file, src_pose, intrinsic in zip(src_rgb_files, src_poses, src_intrinsics):
        src_rgb , src_depth, src_alpha, src_camera = \
        read_image(src_rgb_file, src_pose, 
            intrinsic, max_depth=max_depth, resize_factor=resize_factor)

        src_rgbs.append(src_rgb)
        src_depths.append(src_depth)
        src_alphas.append(src_alpha)
        src_cameras.append(src_camera)
    
    src_alphas = torch.stack(src_alphas, axis=0)
    src_depths = torch.stack(src_depths, axis=0)
    src_rgbs = torch.stack(src_rgbs, axis=0)
    src_cameras = torch.stack(src_cameras, axis=0)
    src_rgbs = src_alphas[..., None] * src_rgbs + (1-src_alphas)[..., None]

    return {
        "rgb": src_rgbs[..., :3],
        "camera": src_cameras,
        "depth": src_depths,
        "alpha": src_alphas,
    }


def read_image(rgb_file, pose, intrinsic_, max_depth, resize_factor=1., white_bkgd=True):
    #rgb.shape = (H, W, 3), depth.shape = (H, W), alpha.shape = (H, W)
    rgb = torch.from_numpy(imageio.imread(rgb_file).astype(np.float32) / 255.0)
    depth = torch.from_numpy(imageio.imread(rgb_file[:-7]+'depth.png').astype(np.float32) / 255.0 * max_depth)
    alpha = torch.from_numpy(imageio.imread(rgb_file[:-7]+'alpha.png').astype(np.float32) / 255.0)
    # rgb = torch.from_numpy(rgb_file)
    # alpha2 = torch.ones_like(rgb[..., 0],dtype=torch.float32)
    # depth = 1 * torch.ones_like(alpha2,dtype=torch.float32)

    image_size = rgb.shape[:2]
    intrinsic = np.eye(4,4)
    intrinsic[:3,:3] = intrinsic_

    if resize_factor != 1:
        image_size = image_size[0] * resize_factor, image_size[1] * resize_factor 
        intrinsic[:2,:3] *= resize_factor
        resize_fn = lambda img, resize_factor: F.interpolate(
                img.permute(0, 3, 1, 2), scale_factor=resize_factor, mode='bilinear',
            ).permute(0, 2, 3, 1)
        
        rgb = rearrange(resize_fn(rearrange(rgb, 'h w c -> 1 h w c'), resize_factor), '1 h w c -> h w c')
        depth = rearrange(resize_fn(rearrange(depth, 'h w -> 1 h w 1'), resize_factor), '1 h w 1 -> h w')
        alpha = rearrange(resize_fn(rearrange(alpha, 'h w -> 1 h w 1'), resize_factor), '1 h w 1 -> h w')

    camera = torch.from_numpy(np.concatenate(
        (list(image_size), intrinsic.flatten(), pose.flatten())
    ).astype(np.float32))
    
    if white_bkgd:
        rgb = alpha[..., None] * rgb + (1-alpha2)[..., None]

    return rgb, depth, alpha, camera

def load_tiny_nerf_data(path, resize_factor=1.0, device='cpu'):
    """
    加载 tiny_nerf_data.npz 格式的数据，并转换为与 read_all 函数相同的输出格式。
    
    参数:
        path: tiny_nerf_data.npz 文件的路径
        resize_factor: 图像调整大小的因子
        device: 张量放置的设备
        
    返回值:
        包含以下键的字典:
        - rgb: 形状为 [N, H, W, 3] 的 RGB 图像张量
        - camera: 形状为 [N, 34] 的相机参数张量
        - depth: 默认深度值
        - alpha: 默认不透明度值
    """
    import numpy as np
    import torch
    
    # 加载 npz 文件
    data = np.load(path)
    images = data['images']  # [N, H, W, 3] RGB 图像
    poses = data['poses']    # [N, 4, 4] 相机到世界的变换矩阵
    focal = float(data['focal'])  # 焦距
    
    # 获取图像尺寸
    N, H, W, C = images.shape
    print(f"原始图像形状: {images.shape}")
    
    # RGB 图像
    rgbs = images
    
    # 调整图像大小
    if resize_factor != 1.0:
        new_H = int(H * resize_factor)
        new_W = int(W * resize_factor)
        
        # 使用 PyTorch 调整大小
        # 将 numpy 转为 torch 进行调整大小
        rgbs_torch = torch.from_numpy(rgbs).float().permute(0, 3, 1, 2)  # [N, 3, H, W]
        
        # 调整大小
        rgbs_torch = F.interpolate(rgbs_torch, size=(new_H, new_W), mode='bilinear', align_corners=False)
        
        # 调整回原始格式
        rgbs = rgbs_torch.permute(0, 2, 3, 1).numpy()  # [N, H', W', 3]
        
        # 更新尺寸和焦距
        H, W = new_H, new_W
        focal = focal * resize_factor
    
    # 创建相机参数数组
    cameras = []
    for pose in poses:
        #把blender的坐标系转换为opencv的坐标系
        c2w = pose
        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)
        # 创建内参矩阵
        intrinsic = np.eye(4)
        intrinsic[0, 0] = focal  # fx
        intrinsic[1, 1] = focal  # fy
        intrinsic[0, 2] = W / 2  # cx
        intrinsic[1, 2] = H / 2  # cy
        
        # 按照 data_utils.read_image 的格式创建相机参数
        camera = np.concatenate(
            ([H, W], intrinsic.flatten(), c2w_opencv.flatten())
        ).astype(np.float32)
        cameras.append(camera)
    
    # 转换为 PyTorch 张量
    rgbs_torch = torch.from_numpy(rgbs).float().to(device)

    cameras_torch = torch.from_numpy(np.array(cameras)).float().to(device)
    
    # 创建对应大小的全1 alpha 通道
    alphas_torch = torch.ones(N, H, W, device=device,dtype=torch.float32)
    
    # 创建深度通道
    depths_torch = torch.ones_like(alphas_torch, device=device,dtype=torch.float32)
    
    return {
        "rgb": rgbs_torch,  # [N, H, W, 3]
        "camera": cameras_torch,  # [N, 34]
        "depth": depths_torch,  # [N, H, W]
        "alpha": alphas_torch,  # [N, H, W]
    }
