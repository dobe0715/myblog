```python
import sys
sys.version
```




    '3.10.11 (main, Apr  5 2023, 14:15:10) [GCC 9.4.0]'



<img src="https://drive.google.com/uc?id=18wzt4KxmNxVLBaaovJZEmF8VC31206vG
">


```python
!git clone https://github.com/ermongroup/SDEdit.git
%cd /content/SDEdit
```

    Cloning into 'SDEdit'...
    remote: Enumerating objects: 156, done.[K
    remote: Counting objects: 100% (155/155), done.[K
    remote: Compressing objects: 100% (89/89), done.[K
    remote: Total 156 (delta 69), reused 129 (delta 57), pack-reused 1[K
    Receiving objects: 100% (156/156), 37.39 MiB | 7.62 MiB/s, done.
    Resolving deltas: 100% (69/69), done.
    /content/SDEdit
    


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
import numpy as np
import matplotlib.pyplot as plt

img_real = plt.imread("/content/drive/MyDrive/Capstone/sde_images/mars_man.png")
img_stroke = plt.imread("/content/drive/MyDrive/Capstone/sde_images/mars_man_stroke.png")

img_mask = img_real - img_stroke[:, :, :3]
img_mask = (1,) - np.array(img_mask != 0, dtype=float)
```


```python
img_mask.shape
```




    (438, 438, 3)




```python
# import torch

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# url = "/content/drive/MyDrive/Capstone/celeba_hq.ckpt"

# ckpt = torch.load(url, map_location=device)
```


```python
import argparse

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
```


```python
# from models.diffusion import Model
# import yaml


# config = "/content/SDEdit/configs/celeba.yml"

# with open(config, 'r') as f:
#     config = yaml.safe_load(f)
#     new_config = dict2namespace(config)

# model = Model(new_config)
# model.load_state_dict(ckpt)
```




    <All keys matched successfully>




```python
# def image_editing_sample(self):
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     url = "/content/drive/MyDrive/Capstone/celeba_hq.ckpt"
#     ckpt = torch.load(url, map_location=device)

#     config = "/content/SDEdit/configs/celeba.yml"

#     with open(config, 'r') as f:
#         config = yaml.safe_load(f)
#         new_config = dict2namespace(config)

#     model = Model(new_config)
#     model.load_state_dict(ckpt)
#     model.to(self.device)
#     model = torch.nn.DataParallel(model)
#     print("Model loaded")
#     ckpt_id = 0
```


```python
import yaml
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision
import torchvision.utils as tvu

from models.diffusion import Model


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def image_editing_denoising_step_flexible_mask(x, t, *,
                                               model,
                                               logvar,
                                               betas):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)

    model_output = model(x, t)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)

    logvar = extract(logvar, t, x.shape)
    noise = torch.randn_like(x)
    mask = 1 - (t == 0).float()
    mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
    sample = mean + mask * torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config

        with open(self.config, 'r') as f:
            config = yaml.safe_load(f)
            config = dict2namespace(config)

        self.config = config

        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
            (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    def image_editing_sample(self):
        print("Loading model")
        # if self.config.data.dataset == "LSUN":
        #     if self.config.data.category == "bedroom":
        #         url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
        #     elif self.config.data.category == "church_outdoor":
        #         url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        # elif self.config.data.dataset == "CelebA_HQ":
        #     url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        # else:
        #     raise ValueError

        # model = Model(self.config)
        # ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
        # model.load_state_dict(ckpt)
        # model.to(self.device)
        # model = torch.nn.DataParallel(model)
        # print("Model loaded")
        # ckpt_id = 0

        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        url = "/content/drive/MyDrive/Capstone/celeba_hq.ckpt"
        ckpt = torch.load(url, map_location=self.device)

        config = "/content/SDEdit/configs/celeba.yml"

        # with open(config, 'r') as f:
        #     config = yaml.safe_load(f)
        #     new_config = dict2namespace(config)

        # self.config = new_config
        
        model = Model(self.config)
        model.load_state_dict(ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        print("Model loaded")
        ckpt_id = 0



        # download_process_data(path="colab_demo")
        # n = self.config.sampling.batch_size
        n = 2
        model.eval()
        print("Start sampling")
        with torch.no_grad():
            name = self.args.npy_name
            # [mask, img] = torch.load("colab_demo/{}.pth".format(name))

            resolution = self.config.data.image_size

            convert_tensor = torchvision.transforms.ToTensor()
            convert_size = torchvision.transforms.Resize((resolution, resolution))

            mask = torch.tensor(name)
            mask = mask.permute(2, 0, 1)
            img = Image.open("/content/drive/MyDrive/Capstone/sde_images/mars_man_stroke.png").convert('RGB')
            img = convert_tensor(img)

            mask = convert_size(mask)
            img = convert_size(img)

            mask = mask.to(self.device)
            img = img.to(self.device)

            img = img.unsqueeze(dim=0)
            img = img.repeat(n, 1, 1, 1)
            # img = np.array(img.cpu(), dtype=np.uint8)
            # x0 = Image.fromarray(img)
            x0 = img

            tvu.save_image(x0, os.path.join(self.args.image_folder, f'original_input.png'))
            x0 = (x0 - 0.5) * 2.

            for it in range(self.args.sample_step):
                e = torch.randn_like(x0)
                total_noise_levels = self.args.t
                a = (1 - self.betas).cumprod(dim=0)
                x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder, f'init_{ckpt_id}.png'))

                with tqdm(total=total_noise_levels, desc="Iteration {}".format(it)) as progress_bar:
                    for i in reversed(range(total_noise_levels)):
                        t = (torch.ones(n) * i).to(self.device)
                        x_ = image_editing_denoising_step_flexible_mask(x, t=t, model=model,
                                                                        logvar=self.logvar,
                                                                        betas=self.betas)
                        x = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                        x[:, (mask != 1.)] = x_[:, (mask != 1.)]
                        # added intermediate step vis
                        if (i - 99) % 100 == 0:
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'noise_t_{i}_{it}.png'))
                        progress_bar.update(1)

                x0[:, (mask != 1.)] = x[:, (mask != 1.)]
                torch.save(x, os.path.join(self.args.image_folder,
                                           f'samples_{it}.pth'))
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'samples_{it}.png'))
```


```python
# import torch
# import torchvision.utils as vutils

# # Create a tensor of shape (3, 64, 64) representing a single RGB image
# img = torch.randn(3, 64, 64)
# img = img.unsqueeze(dim=0)
# img = img.repeat(8, 1, 1, 1)

# # Save the image to a file
# vutils.save_image(img, 'my_image.png')
```


```python
# from PIL import Image
# import torchvision

# convert_tensor = torchvision.transforms.ToTensor()


# img = Image.open("/content/drive/MyDrive/Capstone/sde_images/mars_man_stroke.png")
# img = convert_tensor(img)
# img = torchvision.transforms.Resize((256, 256))(img)
# img = img.unsqueeze(dim=0)

# img.shape
```

    /usr/local/lib/python3.10/dist-packages/torchvision/transforms/functional.py:1603: UserWarning: The default value of the antialias parameter of all the resizing transforms (Resize(), RandomResizedCrop(), etc.) will change from None to True in v0.17, in order to be consistent across the PIL and Tensor backends. To suppress this warning, directly pass antialias=True (recommended, future default), antialias=None (current default, which means False for Tensors and True for PIL), or antialias=False (only works on Tensors - PIL will still use antialiasing). This also applies if you are using the inference transforms from the models weights: update the call to weights.transforms(antialias=True).
      warnings.warn(
    




    torch.Size([1, 4, 256, 256])




```python
import easydict
# import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
# import torch.utils.tensorboard as tb
import copy



def make_parse_args(img_msk):
    args = easydict.EasyDict({'seed': 1234, 
                              'exp': 'exp', 
                              'comment': '', 
                              'verbose': 'info', 
                              'sample': 'store_true', 
                              'i': 'images', 
                              'image_folder': 'images', 
                              'ni': 'store_true', 
                              'npy_name': img_msk, 
                              'sample_step': 3, 
                              't': 400})

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
    args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input("Image folder already exists. Overwrite? (Y/N)")
            if response.upper() == 'Y':
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args
```


```python
config = "/content/SDEdit/configs/celeba.yml"
args = make_parse_args(img_mask)
```

    INFO:root:Using device: cuda
    INFO - <ipython-input-11-20b5aa3c5865> - 2023-06-05 15:13:32,851 - Using device: cuda
    INFO - <ipython-input-11-20b5aa3c5865> - 2023-06-05 15:13:32,851 - Using device: cuda
    


```python
try:
    runner = Diffusion(args, config)
    runner.image_editing_sample()
except Exception:
    logging.error(traceback.format_exc())
```

    Loading model
    Model loaded
    Start sampling
    

    Iteration 0:   0%|          | 0/400 [00:00<?, ?it/s]<ipython-input-10-ffd958c84eb2>:26: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    Iteration 0: 100%|██████████| 400/400 [00:32<00:00, 12.14it/s]
    Iteration 1: 100%|██████████| 400/400 [00:29<00:00, 13.38it/s]
    Iteration 2: 100%|██████████| 400/400 [00:30<00:00, 13.27it/s]
    


```python

```
