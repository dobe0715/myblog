+++
title = "High-Resolution Image Synthesis with Satent Diffusion Models(Stable Diffusion) review"
+++
# High-Resolution Image Synthesis with Satent Diffusion Models(Stable Diffusion)

참고자료
+ paper : https://arxiv.org/abs/2112.10752
+ youtube : https://www.youtube.com/watch?v=rC34475rEnw
+ blog : https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ldm/  




## Contributions
1. 기존의 DM들에 비해 소모되는 컴퓨팅 자원을 훨씬 줄였다. => 탄소배출 감소..^^
2. space에 따라 2-stage로 나눠 학습으로써 더욱 효율적인 모델을 찾을 수 있었다.
3. cross-attention을 U-Net에 적용하여 다양한 class condition을 줄 수 있다.
4. 사전학습모델을 무료공개했다.

## 기존의 이미지 생성모델들

+ GAN
    + 장점 : 좋은 퀄리티의 고해상도 이미지
    + 단점 : 데이터 분포 전체 학습에서의 어려움(mode collapse 위험)

+ VAE(Likelihood-based)
    + 장점 : 데이터 분포 학습이 잘됨
    + 단점 : 셈플 퀄리티 GAN에 비해 상대적 낮음

+ ARM(Auto Regressive Model, pixel RNN,CNN 등)
    + 장점 : 데이터 분포 학습이 잘됨
    + 단점 : sequential 하게 sampling하기 위해 해상도의 한계, 컴퓨팅 요구량

<img src="https://drive.google.com/uc?id=1DdFVj2w-wq8CKrONIGxEJvf91dWvvBDL" height=300>

+ DM(Diffusion Probablistic Model)
    + 장점 : 분포학습 + 퀄리티 + inductive bias
    + 단점 : inference 및 training에서의 cost

+ 2-stage ARM(VQ-VAE, VQ-GAN 등)
    + 장점 : 기존의 해상도 한계를 quantization을 통해 극복
    + 단점 : 모델 표현력을 끌어올리는데 한계가 있다.(compression rate를 높이면, 결국 학습할 파라미터 수가 너무 많아짐)

<img src="https://drive.google.com/uc?id=11VlpjQDLlJakEKBBikgk5IcBHLQkoWKg" height=350>

LDM(Latent Diffusion Model)은, 결국 위의 마지막 두가지를 결합한 모델이다.

## Model

<img src="https://drive.google.com/uc?id=130yyc3D_P9OCcMu4YEtCBWM1w6w2v3Bt" height=300>

두개의 stage로 나뉜다.
1. 이미지 압축 (encoder/decoder)
2. diffusion process

### Perceptual Image Compression
주어진 이미지 x 를 latent space로 mapping해주는 단계이다. pixel 단계에서의 detail한 부분을 날려 압축해주어 계산복잡도를 낮춘다.


+ encoder : $z = \mathcal{E}(x)$
+ decoder : $\tilde{x} = \mathcal{D}(z)$
+ $x$ : $H \times W \times 3$, RGB 이미지
+ $z$ : $h \times w \times c$, latent
+ $f = H/h = W/w$ : downsampling factor. 이 값을 기준으로 LDM 모델 나눔. $f \in \{1, 2, 4, ..., 32\}$

이 때, encoder, decoder를 학습할 때 latent space의 variance를 균등하게 하기 위해서 두가지 regularization loss를 추가한다.  

+ KL-reg : normal distribution과의 KL divergence계산
+ VQ-reg : VQGAN loss와 비슷한 역할.  

<img src="https://drive.google.com/uc?id=1Hp1e3mDsHbQwumhIJEkBW15_ryMGODgY" width=800>

### Latent Diffusion Model
$L_{LDM} := \mathbb{E}_{\mathcal{E}(x), \epsilon \sim N(0, 1), t}\bigg[||\epsilon - \epsilon_\theta(z_t, t)||_2^2\bigg]$

UNet기반 backbone을 사용한다.

### Conditioning Mechanisms
기존의 condition을 줄때에는 단순히 y를 label값으로만 줄 수 있었다. 하지만, 여기서는 Cross-Attention을 이용하여text, resolution, semantic map 등을 domain encoder $\tau_\theta$를 이용하여 embedding해서 UNet에 넣어준다.

$Q = W_Q^{(i)} \cdot \phi_i(z_i), K = W_K^{(i)} \cdot \tau_\theta(y), V = W_V^{(i)} \cdot \tau_\theta(y)$

최종적으로 LDM loss는 다음과 같다.  
$L_{LDM} := \mathbb{E}_{\mathcal{E}(x), y, \epsilon \sim N(0, 1), t}\bigg[||\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y))||_2^2\bigg]$

이 때, $\tau_\theta, \epsilon_\theta$는 동시에 optimize된다. 그리고, condition domain에 따라 $\tau_\theta$를 다양하게 파라미터화 할 수 있다.

## Experiments

### On Perceptual Compression Tradeoffs

downsampling factor $f \in \{1, 2, 4, 8, 16, 32\}$에 따라 $LDM-f$라고 부른다. 또한, $LDM-1$은 기존의 pixel based DM에 대응된다.

<img src="https://drive.google.com/uc?id=137vp-lW-ICBT2mEuN5HfHNGJ1hbdCwAc" height=300>

살펴보면, f=1,2,32일 때 성능이 좋지 않음을 알 수 있다.
즉, tradeoff에 대한 적당한 optimal point가 있다는 뜻

### Image Generation with LDM

<img src="https://drive.google.com/uc?id=1gKJ3dIrbm74nDHRy_G-QSDGZK-wqOAkx" height=400>  
기존 생성모델들의 성능을 거의 다 뛰어넘었다.

<img src="https://drive.google.com/uc?id=1O9Zsjum0gdZasLMGvYz54Mx0WabEVN9z" height=200>  
condition guaidance를 줬을 땐, FID점수도 많이 올릴 수 있다.

<img src="https://drive.google.com/uc?id=1DkkPHLAcCaG_RJTUBP1zmzkBIEAu9Fyx" height=150>  
Super Resolution, Inpainting 분야에서도 좋은 성능을 보인다.

<img src="https://drive.google.com/uc?id=1N5fOTht4zWHiVCcVI0Od5efviGlnboxv" width=500>

## Limitation
computation 요구량을 확실히 줄였지만, 아직은 GAN에 비하면 턱없이 느리다.  
또한, $f = 4$에서 좋은 퀄리티를 보였지만, pixel space에서 정확도를 얻기 위해서는 bottleneck이 될 수 있다.


```python

```
