"""Day proxy of the night query images with an instruction-based image editor: Qwen-Image-Edit-2509 (20B, Apache-2.0)
+ Lightning 4-step LoRA (lightx2v). The instruction asks for midday sunlight with every light source switched off and
the geometry unchanged. Env `qwenedit` (GPU, run in tmux). Weights are loaded straight to the GPU (CPU RAM cap 20 GB).

Output layout as scripts/day_translate.py: <out>/images/<cam>/<original file name>.jpg (resized back to 1506x728) and
<out>/images/<cam>.csv. odo.online / scripts/exp24_day.py read it as images 'day_qwen'.

    python scripts/day_translate_qwen.py --lap L2 --out data/day_proxy/L2_qwen [--limit N] [--cams fl fr] [--steps 4]
"""
import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f'{ROOT}/data/extracted'
CAMS = ['fl', 'fr', 'cl', 'cr', 'r', 'rl', 'rr']
LAP_STARTS = [0.0, 185.456, 365.569, 523.375, 633.972, 752.192, 896.116, 1007.673, 1120.703, 1194.393]
MODEL = 'Qwen/Qwen-Image-Edit-2509'
PROMPT = ('Change this night photo into the same photo taken at midday on a sunny day: bright natural daylight, clear '
          'blue sky, sunlit asphalt and buildings with realistic sun shadows. All floodlights, street lamps, stadium '
          'lights, signs and windows are switched off and dark, no glow, no lens flare. Keep exactly the same camera '
          'viewpoint, framing and geometry: the same buildings, grandstands, fences, barriers, poles, kerbs, track '
          'markings, cars and car body, unchanged in position and shape. Photorealistic onboard race-car camera image.')


def load_pipe(steps):
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline, QwenImageTransformer2DModel
    from huggingface_hub import hf_hub_download
    from transformers import Qwen2_5_VLForConditionalGeneration
    tr = QwenImageTransformer2DModel.from_pretrained(MODEL, subfolder='transformer', torch_dtype=torch.bfloat16,
                                                     device_map='cuda')
    te = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, subfolder='text_encoder', torch_dtype=torch.bfloat16,
                                                            device_map='cuda')
    # scheduler settings of the Lightning recipe (lightx2v/Qwen-Image-Lightning)
    sched = FlowMatchEulerDiscreteScheduler.from_config(dict(
        base_image_seq_len=256, base_shift=math.log(3), invert_sigmas=False, max_image_seq_len=8192,
        max_shift=math.log(3), num_train_timesteps=1000, shift=1.0, shift_terminal=None, stochastic_sampling=False,
        time_shift_type='exponential', use_beta_sigmas=False, use_dynamic_shifting=True, use_exponential_sigmas=False,
        use_karras_sigmas=False))
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, transformer=tr, text_encoder=te, scheduler=sched,
                                                     torch_dtype=torch.bfloat16)
    pipe.to('cuda')
    lora = hf_hub_download('lightx2v/Qwen-Image-Lightning',
                           f'Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-{steps}steps-V1.0-bf16.safetensors')
    pipe.load_lora_weights(lora)
    pipe.fuse_lora()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lap', default='L2'); ap.add_argument('--out', required=True)
    ap.add_argument('--cams', nargs='+', default=CAMS); ap.add_argument('--limit', type=int)
    ap.add_argument('--t0', type=float, help='start offset into the lap [s]')
    ap.add_argument('--steps', type=int, default=4); ap.add_argument('--long_side', type=int, default=1024)
    a = ap.parse_args()
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as qp
    gt0 = pd.read_csv(f'{DATA}/gt.csv', nrows=1).timestamp_ns.iloc[0]
    i = int(a.lap[1:]); ta, tb = LAP_STARTS[i], LAP_STARTS[i + 1]
    if a.t0 is not None:
        ta = LAP_STARTS[i] + a.t0
    pipe = load_pipe(a.steps)
    pool = ThreadPoolExecutor(8)
    for cam in a.cams:
        f = pd.read_csv(f'{DATA}/images/{cam}.csv')
        t = (f.stamp_ns - gt0) / 1e9
        f = f[(t >= ta - 0.5) & (t < tb + 0.5)].reset_index(drop=True)
        if a.limit:
            f = f.iloc[:a.limit]
        os.makedirs(f'{a.out}/images/{cam}', exist_ok=True)
        todo = [r for r in f.itertuples() if not os.path.exists(f'{a.out}/{r.file}')]
        tic = time.time(); writes = []
        for n, (r, im) in enumerate(zip(todo, pool.map(lambda r: cv2.imread(f'{DATA}/{r.file}'), todo))):
            H, W = im.shape[:2]
            # both sides multiples of 32: the pipeline snaps the condition image to multiples of 32, and any mismatch with
            # the output size shifts / stretches the result (13.6 px median displacement at 1024x496). 1024x512 stretches
            # the image by 3 % vertically; resizing back to W x H undoes that exactly.
            w = a.long_side; h = int(round(H * w / W / 32)) * 32
            qp.VAE_IMAGE_SIZE = w * h                # condition latents at the output resolution (default 1 MP)
            x = Image.fromarray(cv2.cvtColor(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            with torch.inference_mode():
                y = pipe(image=[x], prompt=PROMPT, negative_prompt=' ', true_cfg_scale=1.0, num_inference_steps=a.steps,
                         height=h, width=w, generator=torch.Generator('cuda').manual_seed(0)).images[0]
            y = cv2.resize(cv2.cvtColor(np.asarray(y), cv2.COLOR_RGB2BGR), (W, H), interpolation=cv2.INTER_CUBIC)
            writes.append(pool.submit(cv2.imwrite, f'{a.out}/{r.file}', y, [cv2.IMWRITE_JPEG_QUALITY, 95]))
            if n % 100 == 0:
                print(cam, n, len(todo), f'{(time.time() - tic) / (n + 1) * 1000:.0f} ms/img', flush=True)
        for w_ in writes:
            w_.result()
        f.to_csv(f'{a.out}/images/{cam}.csv', index=False)
        print(cam, 'done', len(todo), f'{time.time() - tic:.0f} s', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
