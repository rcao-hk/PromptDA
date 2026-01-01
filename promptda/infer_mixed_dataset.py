import os

import argparse
import cv2
import glob
import matplotlib
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from promptda.core import PromptDA

import time
def benchmark_inference(model,
                          image,
                          raw,
                          device='cuda',
                          n_warmup=10,
                          n_iters=50):
    """
    精确测 model.infer_image 的推理时间（包含 image2tensor + forward + upsample + .cpu().numpy()）

    model: 带 infer_image 方法的模型实例
    raw_image: infer_image 的 raw_image 输入（一般是 HxWx3 的 numpy 或 PIL）
    depth_low_res: infer_image 的 depth_low_res 输入
    input_size: 传给 infer_image 的 input_size
    device: 'cuda' 或 'cpu'
    n_warmup: 预热次数（不计时）
    n_iters: 正式计时次数
    """
    assert device in ['cuda', 'cpu']

    torch.set_grad_enabled(False)

    # 如果 image2tensor / forward 里自己做 .to(device)，这里就不用动 raw_image / depth_low_res。
    # 如果 forward 依赖外面传入已经在 GPU 的 tensor，就需要你在 infer_image 里加 .to(device)。

    # 要计时的函数（封装一下，方便统一处理）
    def _run_once():
        # 返回值不关心，只是触发一次完整推理
        _ = model(image, raw)

    # ---------- warm-up ---------- #
    for _ in range(n_warmup):
        _run_once()
    if device == 'cuda':
        torch.cuda.synchronize()

    times_ms = []

    # ---------- 正式计时 ---------- #
    if device == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)

        for _ in range(n_iters):
            starter.record()
            _run_once()
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))  # ms
    else:
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _run_once()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    times = torch.tensor(times_ms)
    mean_ms = times.mean().item()
    std_ms  = times.std(unbiased=False).item()

    print(f"[{device}] infer_image, {n_iters} runs")
    print(f"  mean : {mean_ms:.3f} ms / image")
    print(f"  std  : {std_ms:.3f} ms")
    print(f"  FPS  : {1000.0 / mean_ms:.2f}")

    return mean_ms, std_ms, times_ms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    parser.add_argument('--dataset', type=str, default='PhoCAL', choices=['HAMMER', 'HouseCat6D', 'PhoCAL', 'TransCG', 'XYZ-IBD', 'YCB-V', 'T-LESS', 'GN-Trans', 'ROBI'])
    parser.add_argument('--dataset_root', type=str, default='/data/robotarm/dataset')
    parser.add_argument('--split', type=str, default='/home/robotarm/object_depth_percetion/dataset/splits/PhoCAL_test.txt', help='Path to split file listing RGB images')
    parser.add_argument('--output_root', type=str, default='/data/robotarm/result/depth/mixed')
    parser.add_argument('--method', type=str, default='d3roma_zs_360x640')
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["vits", "vitb", "vitl", "vitg"],
        default="vitl",
        help="Model encoder type",
    )
    parser.add_argument('--img_size', type=int, default=518)
    parser.add_argument('--min-depth', type=float, default=0.001)
    parser.add_argument('--max-depth', type=float, default=5)
    parser.add_argument('--camera', type=str, default='d435', choices=['l515', 'd435', 'tof'])
    args = parser.parse_args()

    args.pred_only = True
    args.grayscale = True
    depth_factor = 1000.0
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model = PromptDA.from_pretrained("depth-anything/prompt-depth-anything-{}".format(args.encoder)).to(DEVICE).eval()

    rgb_trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.ConvertImageDtype(torch.float32),
                                    transforms.Resize((args.img_size, args.img_size), transforms.InterpolationMode.BICUBIC)])
    depth_trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.ConvertImageDtype(torch.float32)])

    depth_resize = transforms.Resize((args.img_size, args.img_size), transforms.InterpolationMode.NEAREST)
    # ===== 读取 split 文件中的 rgb 路径 =====
    with open(args.split, 'r') as f:
        rgb_lines = [line.strip().split()[0] for line in f if line.strip()]

    for rgb_rel_path in tqdm(rgb_lines):
        rgb_path = os.path.join(args.dataset_root, args.dataset, rgb_rel_path)
        depth_scale = 1.0
        # 推导出 raw depth 路径
        if args.dataset == 'HAMMER':
            scene_name = rgb_rel_path.split('/')[0]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, scene_name, 'polarization', f'depth_{args.camera}', f'{frame_id:06d}.png')
        elif args.dataset == 'HouseCat6D':
            scene_name = rgb_rel_path.split('/')[0]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, scene_name, 'depth', f'{frame_id:06d}.png')
        elif args.dataset == 'PhoCAL':
            scene_name = rgb_rel_path.split('/')[0]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, scene_name, 'depth', f'{frame_id:06d}.png')
        elif args.dataset == 'TransCG':
            scene_name = rgb_rel_path.split('/')[1]
            frame_id = int(rgb_rel_path.split('/')[-2])
            if args.camera == 'd435':
                depth_path = os.path.join(args.dataset_root, args.dataset, 'scenes', scene_name, f'{frame_id}', 'depth1.png')
            elif args.camera == 'l515':
                depth_path = os.path.join(args.dataset_root, args.dataset, 'scenes', scene_name, f'{frame_id}', 'depth2.png')
        elif args.dataset == 'GN-Trans':
            scene_name = rgb_rel_path.split('/')[1]
            frame_id = int(rgb_rel_path.split('/')[-1].split('_')[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, 'scenes', scene_name, f'{frame_id:04d}_depth_sim.png')
        elif args.dataset == 'XYZ-IBD':
            depth_scale = 0.09999999747378752
            scene_name = rgb_rel_path.split('/')[1]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, 'val', scene_name, 'depth_xyz', f'{frame_id:06d}.png')
        elif args.dataset == 'YCB-V':
            depth_scale = 0.1
            scene_name = rgb_rel_path.split('/')[1]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, 'test', scene_name, 'depth', f'{frame_id:06d}.png')
        elif args.dataset == 'T-LESS':
            depth_scale = 0.1
            scene_name = rgb_rel_path.split('/')[1]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, 'test_primesense', scene_name, 'depth', f'{frame_id:06d}.png')
        elif args.dataset == 'ROBI':
            depth_scale = 0.03125
            scene_name = rgb_rel_path.split('/')[1] + '_' + rgb_rel_path.split('/')[2]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0].split('_')[-1])
            depth_path = os.path.join(args.dataset_root, args.dataset, rgb_rel_path.replace("Stereo", "Depth").replace('LEFT_', 'DEPTH_').replace('.bmp', '.png'))
            
        if not os.path.exists(depth_path):
            print(f'[Warning] Raw depth not found: {depth_path}, skipping')
            continue

        try:
            raw = np.array(Image.open(depth_path)) * depth_scale * 1e-3
            rgb = np.array(Image.open(rgb_path))
            # 兼容灰度图：(H, W) -> (H, W, 3)
            if rgb.ndim == 2:
                rgb = np.repeat(rgb[..., None], 3, axis=2)

            # 兼容单通道：(H, W, 1) -> (H, W, 3)
            elif rgb.ndim == 3 and rgb.shape[2] == 1:
                rgb = np.repeat(rgb, 3, axis=2)

            # 兼容 RGBA：(H, W, 4) -> (H, W, 3)
            elif rgb.ndim == 3 and rgb.shape[2] == 4:
                rgb = rgb[..., :3]

            # 兜底：其它异常 shape 直接报错
            elif rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"Unexpected rgb shape: {rgb.shape}")

            raw_height, raw_width = rgb.shape[:2]
            
        except Exception as e:
            print(f"[Error] Failed to read {rgb_path} or {depth_path}: {e}")
            continue

        if args.dataset in ['TransCG']:
            save_dir = os.path.join(args.output_root, args.dataset, args.camera, args.method, args.encoder, scene_name)
        else:
            save_dir = os.path.join(args.output_root, args.dataset, args.method, args.encoder, scene_name)
        os.makedirs(save_dir, exist_ok=True)
        
        rgb = rgb_trans(rgb).unsqueeze(0).to(DEVICE)
        raw = depth_trans(raw).unsqueeze(0).to(DEVICE)
        
        # print(rgb.shape)
        benchmark_inference(model, rgb, raw)
        # pred_depth = model.predict(rgb, raw) # HxW, depth in meters

        # depth_recover = transforms.Resize((raw_height, raw_width), transforms.InterpolationMode.NEAREST)
        # pred_depth = depth_recover(pred_depth)
        # pred_depth = pred_depth.squeeze(0).squeeze(0).detach().cpu().numpy()
        
        # metric_depth = (pred_depth * depth_factor).astype(np.uint16)
        # cv2.imwrite(os.path.join(save_dir, f'{frame_id:06d}_depth.png'), metric_depth)
