import torch
import numpy as np
import gaussian_splatting.utils as utils
from gaussian_splatting.trainer import Trainer
import gaussian_splatting.utils.loss_utils as loss_utils
from gaussian_splatting.utils.data_utils import read_all,load_tiny_nerf_data
from gaussian_splatting.utils.camera_utils import to_viewpoint_camera
from gaussian_splatting.utils.point_utils import get_point_clouds
from gaussian_splatting.gauss_model import GaussModel
from gaussian_splatting.gauss_render import GaussRenderer

import contextlib

from torch.profiler import profile, ProfilerActivity

USE_GPU_PYTORCH = True
USE_PROFILE = False

class GSSTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs.get('data')
        self.gaussRender = GaussRenderer(**kwargs.get('render_kwargs', {}))
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.0
    
    def on_train_step(self):
        ind = np.random.choice(len(self.data['camera']))
        camera = self.data['camera'][ind]
        rgb = self.data['rgb'][ind]
        depth = self.data['depth'][ind]
        mask = (self.data['alpha'][ind] > 0.5)
        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)

        if USE_PROFILE:
            prof = profile(activities=[ProfilerActivity.CUDA], with_stack=True)
        else:
            prof = contextlib.nullcontext()

        with prof:
            out = self.gaussRender(pc=self.model, camera=camera)

        if USE_PROFILE:
            print(prof.key_averages(group_by_stack_n=True).table(sort_by='self_cuda_time_total', row_limit=20))


        l1_loss = loss_utils.l1_loss(out['render'], rgb)
        ssim_loss = 1.0-loss_utils.ssim(out['render'], rgb)

        total_loss = (1-self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss
        psnr = utils.img2psnr(out['render'], rgb)
        log_dict = {'total': total_loss,'l1':l1_loss, 'ssim': ssim_loss, 'psnr': psnr}

        return total_loss, log_dict

    def on_evaluate_step(self, **kwargs):
        import matplotlib.pyplot as plt
        ind = np.random.choice(len(self.data['camera']))
        camera = self.data['camera'][ind]
        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)
        # 获取原始图像
        rgb = self.data['rgb'][ind].detach().cpu().numpy()
        
        # 获取原始图像尺寸
        original_height, original_width = rgb.shape[:2]
        
        # 对GaussRenderer进行渲染
        out = self.gaussRender(pc=self.model, camera=camera)
        rgb_pd = out['render'].detach().cpu().numpy()
        
        # 调整渲染图像大小以匹配原始图像
        from skimage.transform import resize
        if rgb.shape != rgb_pd.shape:
            rgb_pd = resize(rgb_pd, (original_height, original_width), anti_aliasing=True, preserve_range=True)
            # 确保数据类型相同
            rgb_pd = rgb_pd.astype(rgb.dtype)
        
        # 水平拼接图像
        image = np.concatenate([rgb, rgb_pd], axis=1)
        utils.imwrite(str(self.results_folder / f'image-{self.step}.png'), image)

if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="True"
    device = 'cuda'
    folder = './tiny_nerf_data.npz'
    data = load_tiny_nerf_data(folder, resize_factor=1.0,max_depth=5.0,device=device)
    data = {k: v.to(device) for k, v in data.items()}
    data['depth_range'] = torch.Tensor([[1,3]]*len(data['rgb'])).to(device)
    print(data['depth'].shape)

    points = get_point_clouds(data['camera'], data['depth'], data['alpha'], data['rgb'])
    print(points.coords.shape)
    raw_points = points.random_sample(2**14)
    print(raw_points.coords.shape)
    # raw_points.write_ply(open('points.ply', 'wb'))

    gaussModel = GaussModel(sh_degree=4, debug=False)
    gaussModel.create_from_pcd(pcd=raw_points)
    
    render_kwargs = {
        'white_bkgd': False,
    }
    results_folder = 'tiny_result/test'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    trainer = GSSTrainer(model=gaussModel, 
        data=data,
        train_batch_size=1, 
        train_num_steps=5000,
        i_image =100,
        train_lr=1e-3, 
        amp=False,
        fp16=False,
        results_folder=results_folder,
        render_kwargs=render_kwargs,
    )

    trainer.on_evaluate_step()
    trainer.train()