+++
title = 'Denoising Diffusion Probabilistic Models(DDPM) reivew'
+++

# Denoising Diffusion Probabilistic Models
참고 링크
+ Diffusion Model 수학이 포함된 tutorial : https://www.youtube.com/watch?v=uFoGaIVHfoE
+ learn open cv에서 올린 포스트 : https://learnopencv.com/denoising-diffusion-probabilistic-models/

Paper
+ https://arxiv.org/pdf/2006.11239.pdf

#### 논문에서의 아이디어
+ 데이터에 매우 작은 노이즈를 추가하는 process를 중첩해서 해주면 normal distribution 까지 보낼 수 있다.
+ 그리고 이것을 마찬가지로 순차적으로 denoising 해줘 이미지를 복원할 수 있다.

#### 해야할 것
1. forward : noise의 중첩을 어떻게 표현할 것인지?
2. backward : noising과정을 어떻게 파라미터화 해서 loss를 구하고 backpropa할 것인가?

#### 어떻게?
1. markov chain을 가정하여 noise의 중첩을 설명

2. variational inference를 이용해 noising을 설명하고, reparameter화 하여서 loss를 구성하였다
    + VI란, 구하고자하는 posterior를 계산할 수 없어 간접적으로 알고있는 분포를 통해 lower bound를 최대화 하는 방법이다.



<img src="https://drive.google.com/uc?id=1DzU3eReh-eNBS8GsO7tVU5hfAuhdqhnj" width=300>  

이제, 그 어떻게에 대해 좀더 자세하게 알아보자

<img src="https://drive.google.com/uc?id=19lhFdBzXRkc7Z9_0wOffcPXxnfJk0ZL8">  




#### 수식 정리
+ $x_0$ : 실제 이미지
+ $x_t$ : t번의 노이즈가 중첩 된 이미지($x_T$는 nomal distribution이 된다고 가정)
+ $q(x_t|x_{t-1})$ : forward process, 스케쥴링된 분산값인 $\beta_t$들에 의해 변화됨.
+ $q(x_{t-1}|x_t)$ : q로부터 유도된 denoising함수
+ $p_\theta(x_{t-1}|x_t)$ : 파라미터 $\theta$를 통해 근사된 denoising함수

## 1. Forward process
(목표 : noise 과정을 잘 설명하기)


+ $q(x_t|x_{t-1}) = N(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t 𝐈)$

+ 이와같이 구성하게 되면 forward process를 효과적으로 표현할 수 있고, 나아가서 reparameterization trick을 통해 backproba에도 사용할 수 있다.
+ 이것을 중첩하게 되면 어떻게 되는가?

$\alpha_t=1-\beta_t$  
$\bar{\alpha_t}=\alpha_t*\alpha_{t-1}*\cdot\cdot\cdot*\alpha_1$  
라고 했을 때, 다음이 성립한다.
+ $q(x_t|x_0) = N(x_t;\sqrt{\bar{\alpha_t}}x_0, (1-\bar{\alpha_t})I)$


(수식증명)  
<img src="https://drive.google.com/uc?id=18eBqVlXZZAVebxmDT4a0ijMdGRrx9hWg" width=500>

## 2. Backward process with Loss

우선, 전통적인 diffusion에 대하여 variational inference를 통한 loss 텀이 다음과 같이 있다.

+ $\mathbb{E}_q\bigg[D_{KL}(q(x_T|x_0) || p(x_T)) + \sum_{t>1}{D_{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t)) - \log(p_\theta(x_0|x_1))}\bigg]$

왼쪽부터, $L_T$, $L_{t-1}$, $L_0$라고 이야기한다.
+ $L_T$ : 마지막 결과에 대한 regulerization term이라고 볼 수 있다. 하지만, 여기서는 $\beta_t$들을 통해 스캐쥴링되어있어 결과적으로 나오는 값과 실제 가우스분포와의 차이를 계산하면 항상 동일하게 나와서 상수로 일정해진다. 따라서, 생략이 가능하다.
+ $L_0$ : reverse 마지막 하고나서 실제이미지와의 reconstruction term이다. 논문에서는 해당 부분을 lossless codelength라는 표현을 하며 큰 영향력이 없는 loss여서 이부분도 생락했다고 한다. (사실 마지막에 노이즈 엄청 조금은 있으나 마나하긴하다 실제로 코드에서도 sampling할 때 마지막 term은 그대로 보냈다.)

이제, 중간의 $L_{t-1}$을 자세히 뜯어보자. 여기를 계산하기 위해 두가지를 알아야한다.

1. $q(x_{t-1}|x_t, x_0)$
2. $p_\theta(x_{t-1}|x_t)$

__q구하기__

먼저, $x_{t-1}$을 $x_t, x_0$에서 비롯된 평균 분산으로부터 sampling하는 것이라고 생각하면, 다음의 식을 구성할 수 있다.

(여기선 분산은 스케줄된 것을 이용한다)  
$q(x_{t-1}|x_t, x_0) = N(x_{t-1};\tilde{\mu_t}(x_t, x_0), \tilde{\beta_t}I)$  


(where)  
$\tilde{\mu_t}(x_t, x_0) = \cfrac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_t}\beta_t x_0 + \cfrac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$  
$\tilde{\beta_t} = \cfrac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$

q로부터 유도되어, $x_t$에서 $x_0$까지 얼만큼 interpolation한 곳에서 sampling할 것인가 라고 의미적으로 이해할 수 있다

__p구하기__

그러면, 이 값을 주어진 파라미터 $\theta$로 함수 $p_\theta(\cdot)$를 어떻게 구성할 것인가가 중요한 문제이다.

$p_\theta(x_{t-1}|x_t) = N(x_{t-1};\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$이와 같이 평균과 분산에 대해 파라미터화 할 수 있다.

여기에서도 앞과같이 분산은 학습하지않고 평균만 학습한다고 하면,
+ $\Sigma_\theta(x_t, t) = \sigma_t^2I$

그리고,
+ $\sigma_t^2 = \tilde{\beta_t} = \cfrac{1-\tilde{\alpha}_{t-1}}{1-\tilde{\alpha}_t} \beta_t$
+ $\sigma_t^2 = \beta_t$

다음의 관계가 성립함을 실험적으로 알려져 있다. 그러면, 분산을 같게 했으니  
 이제 이 평균을 어떻게 생각하는지가 관건이다. 결국 각각의 평균만 잘 같아지도록 하면된다!!

(수식 유도)  
<img src="https://drive.google.com/uc?id=18PvMsHAh4VvKUIQlOFmiqwYDLXZlKnTd" width=600>  
<img src="https://drive.google.com/uc?id=1ga4TFfOKed84ZWZMnYQhZEGzJBNDNWWh" width=550>  


이 논문에서 나름 키포인트라고 생각함. 평균을 학습하는 것이아니라, 잔차 ($\epsilon$을 학습하는 방향으로 유도한다.)   
그래서, $\tilde{\mu}(x_t, x_0)$ term에서, 이 $x_0$대신에 밑에와 같이 reparameterization trick을 사용할 때의 입실론 값을 학습하고자하는 값으로 잡은 것이다!

<img src="https://drive.google.com/uc?id=1vEHzIul1idE9IV0bcO0fZR99-fHgpxd1" width=550>  
<img src="https://drive.google.com/uc?id=1OPetvGvtNxf5Wn6aZOYMQaQ_mtDEHZFn" width=600>

최종적으로 단순화한 loss는 다음과 같다.
$$L_{simple}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\bigg[||\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)||^2\bigg]$$

추가적으로, 이 논문에선 $\theta$를 학습하기 위해 U-Net을 이용했다고하고, time step t를 잘 임베딩하기 위해 self-attention기술을 이용했다고 한다.

<img src="https://drive.google.com/uc?id=1w1l3-VYiEOGHRewha7adMM8HC-ZkklG8" height=200>

sampling과정에서의 denoising이 결과적으로 보니, score기반 Langevin danamics와 동일하다고도 볼 수 있다.

## 실습했던 코드들


```python
# import cv2
# from datetime import datatime

import gc, os, math, base64, random
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.cuda import amp

import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as TF
import torchvision.datasets as datasets
from torchvision.utils import make_grid


from IPython.display import display, HTML, clear_output
```


```python
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512):
        super().__init__()

        half_dim = time_emb_dims // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

        ts = torch.arange(total_time_steps, dtype=torch.float32)

        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        self.time_blocks = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(in_features=time_emb_dims, out_features=time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dims_exp, out_features=time_emb_dims_exp),
        )

    def forward(self, time):
        return self.time_blocks(time)
```


```python
class AttentionBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels

        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.mhsa = nn.MultiheadAttention(embed_dim=self.channels, num_heads=4, batch_first=True)

    def forward(self, x):
        B, _, H, W = x.shape
        h = self.group_norm(x)
        h = h.reshape(B, self.channels, H * W).swapaxes(1, 2)
        h, _ = self.mhsa(h, h, h)
        h = h.swapaxes(2, 1).view(B, self.channels. H, W)
        return x + h
```


```python
class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels, dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act_fn = nn.SiLU()
        # Group1?
        self.normalize_1 = nn.GroupNorm(num_groups=8, num_channels=self.in_channels)
        self.conv_1 = nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.out_channels, kernel_size=3, padding="same")

        # Group2 time embedding
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=self.out_channels)

        # Group3
        self.normalize_2 = nn.GroupNorm(num_groups=8, num_channels=self.out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_2 = nn.Conv2d(in_channels=self.out_channels,
                                out_channels=self.out_channels, kernel_size=3, padding="same")

        if self.in_channels != self.out_channels:
            self.match_input = nn.Conv2d(in_channels=self.in_channels,
                                         out_channels=out_channels, kernel_size=1, stride=1)
        else:
            self.match_input = nn.Identity()

        if apply_attention:
            self.attention = AttentionBlock(channels=self.out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t):
        # Group1
        h = self.act_fn(self.normalize_1(x))
        h = self.conv_1(h)

        # Group2
        # add in timestep embedding
        h += self.dense_1(self.act_fn(t))[:, :, None, None]    # 2차원짜리 4차원으로(unsqueeze?)

        # Group3
        h = self.act_fn(self.normalize_2(h))
        h = self.dropout(h)
        h = self.conv_2(h)

        return h
```


```python
class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels=channels, out_channels=channels,
                                   kernel_size=3, stride=2, padding=1)

    def forward(self, x, *args):
        return self.downsample(x)

class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                     nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
                                     )

    def forward(self, x, *args):
        return self.upsample(x)
```


```python
class UNet(nn.Module):
    def __init__(
    self,
    input_channels=3,    # RGB 3개에 대한 평균
    output_channels=3,
    num_res_blocks=2,
    base_channels=64,
    base_channels_multiples=(1, 2, 2, 4),
    apply_attention=(False, False, True, False),
    dropout_rate=0.1,
    time_multiple=4,
    ):
        super().__init__()

        time_emb_dims_exp = base_channels * time_multiple
        self.time_embeddings = SinusoidalPositionEmbeddings(time_emb_dims=base_channels,
                                                           time_emb_dims_exp=time_emb_dims_exp)

        self.first = nn.Conv2d(in_channels=input_channels, out_channels=base_channels,
                              kernel_size=3, padding="same")

        num_resolutions = len(base_channels_multiples)

        # encoder of UNet, dimension reduction part
        self.encoder_blocks = nn.ModuleList()
        curr_channels = [base_channels]
        in_channels = base_channels

        for level in range(num_resolutions):
            out_channels = base_channels * base_channels_multiples[level]

            for _ in range(num_res_blocks):

                block = ResnetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate,
                time_emb_dims=time_emb_dims_exp,
                apply_attention=apply_attention[level],
                )
                self. encoder_blocks.append(block)

                in_channels = out_channels
                curr_channels.append(in_channels)

            if level != (num_resolutions - 1):
                self.encoder_blocks.append(DownSample(channels=in_channels))
                curr_channels.append(in_channels)

        # Bottleneck in between
        self.bottleneck_blocks = nn.ModuleList(
            (
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=True,
                ),
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=False,
                ),
            )
        )

        # Decoder in UNet, Dimension restoration with skip-connections ?!
        self.decoder_blocks = nn.ModuleList()

        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * base_channels_multiples[level]

            for _ in range(num_res_blocks + 1):
                encoder_in_channels = curr_channels.pop()     # 이거 맛있다 ㅋㅋ
                block = ResnetBlock(
                    in_channels=encoder_in_channels + in_channels, # ??? -> 아 이거 concat하니까 당연히 더해야지
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                )

                in_channels = out_channels
                self.decoder_blocks.append(block)

            if level != 0:
                self.decoder_blocks.append(UpSample(in_channels))

        self.final = nn.Sequential(
        nn.GroupNorm(num_groups=8, num_channels=in_channels),
        nn.SiLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=output_channels, kernel_size=3,
                 stride=1, padding="same"),
        )
    def forward(self, x, t):

        time_emb = self.time_embeddings(t)

        h = self.first(x)
        outs = [h]

        for layer in self.encoder_blocks:
            h = layer(h, time_emb)
            outs.append(h)

        for layer in self.bottleneck_blocks:
            h = layer(h, time_emb)

        for layer in self.decoder_blocks:
            if isinstance(layer, ResnetBlock):
                out = outs.pop()
                h = torch.cat([h, out], dim=1)    # skip-connection
            h = layer(h, time_emb)

        h = self.final(h)

        return h
```


```python
M = UNet()
```


```python
t = torch.randint(low=1, high=1000, size=(3,))
xt = torch.rand((3,3,64,64))
```


```python
pred_noise = M(xt, t)
```


```python
pred_noise.shape
```




    torch.Size([3, 3, 64, 64])


