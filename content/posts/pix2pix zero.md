# Review : Zero-shot Image-to-Image Translation(pix2pix zero)
+ pre-train된 모델만으로 pix2pix 작업을 완성도 있게 수행한 모델

## 1. 핵심내용
+ __traing free, prompt free__
+ input text prompting 없이 자동으로 direction editing
+ cross-attention guidance를 통해 content 보존
+ auto correlation regularization
+ conditional GAN distillation

## 2. Related Works


### Deep image editing with GANs
+ 기존의 연구들은 target imgae의 latent vector를 disentangle한 space로 보냄으로써, latent vector를 조작하여 image translation을 수행하였다.
    + ex) stylegan에서의 W space
    + single category에 대해 효과적, high quality inversion
    + 단점은 image set을 내가 잘 사전에 준비해놔야한다...  
    + pre trained GAN이용!

<img src="https://drive.google.com/uc?id=1eIEBfFJxMNhBIHp6UidySgTOqleDnsdF
" height=300>

### Text2Image models
+ 기존의 모델들은 text로 입력한 부분 이외의 부분에서도 변경이 많이 일어난다.
    + 이러한 부분을 제어하는 것이 중요한데, mask기법을 이용한 기술이 많이사용되었다.
+ 또한, diffusion기반 translation모델들은 추가학습이 필요하다.
    + ex) Palette, InstructPix2Pix, PITI

+ __pix2pix zero : mask x, 추가학습x__

### Image editing with diffusion models
+ Imagic : 성능은 훌륭 but, 모델에 맞는 fine-tune 필요
+ Prompt2Prompt : cross-attention map 이용. fine-tune 필요 x
+ __pix2pix zero : 성능훌륭, fine-tune 필요 x!__

## 3. Method
+ 목적 : 주어진 image에서 cat -> dog의 변환과정 수행
<img src="https://drive.google.com/uc?id=115u0U73i2FmhGKR-iKEw2v0ab8-hw3cs">

<img src="https://drive.google.com/uc?id=1pUmo8SXOyP-GNd4piybj0SJb2jJtj_1r" width=600>

### Inverting Real Images
+ 사전 준비.
+ image가 $\tilde{x} \in R^{512*512*3}$이면, stable diffusion을 통해 $x_0 \in R^{64*64*4}$ 으로 encoding 해놓고 시작

#### Deterministic inversion
+ DDPM과 DDIM의 가장 큰 차이점은 inversion과정이다.
<img src="https://drive.google.com/uc?id=1kyd3yFFqvVErt5uH8SwM-oGVEI8HXOwQ">
+ ddpm은 마르코프 연쇄 룰 ($q(x_{t}|x_{t-1}, ..., x_0)=q(x_t|x_{t-1})$ "다음것은 오직 바로 직전 것으로 유도된다" 라고 가정하여 inversion 수식유도
+ ddim은 $q(x_{t}|x_{t-1}, x_0)$ "다음것은 처음과 직전 이미지에의해 결정된다"를 가정해서 inversion 수식유도

<img src="https://drive.google.com/uc?id=1R5Gc0t5S4U7uw0v_14cXtcg10OfNtWkX" width=700>

+ $\sigma$에 따라 inversion 모델이 ddpm이 되기도하고 deterministic 해지기도 한다. 여기선 0이라고 하여 inversion
+ $\epsilon$은 각t번째 time step에서의 noise라고 생각할 수 있다.
+ 즉, $x_t$번째 이미지를 통해 처음 이미지와 $t$번째 noise를 추출하여 $\alpha_{t-1}$번째 가중치로 interpolation해서 $x_{t-1}$번째 이미지를 추측한다는 의미

#### Noise regularization

+ 위와같이 DDIM inversion하면서 추출한 Noise map($\epsilon$)이 가우스 distribution을 따르지 않는 경우, image editing이 잘 수행되지 않게 된다. 즉, noise는 다음 조건을 따를 수록 좋다
    1. 임의의 두 위치 pair에 대해, 그 corelation 값이 낮아야 한다
    2. 각 spatial위치에 대해 평균 : 0, 분산 : 1 을 만족해야한다.

+ __Loss__  
$L_{auto} = L_{pair} + \lambda L_{KL}$

+ $L_{pair}$  


<img src="https://drive.google.com/uc?id=1eqgUoCggFH9E3VS5Dx7xl3uv9hthIJA_" width=600>  

+ 보통은 $\delta=1$에 대해서만 계산하지만, 여기선 더 넓은 range에 대해 적용
+ 모든 위치에 대해 안하는건 연산량 너무 많아서인듯.. 대충 생각해도 시간복잡도가 $O(n^4)$가 나온다

+ $L_{KL}$  
    + mean=0, var=1에 너무 fit하게 맞춰버리면, denoise 과정에서 발산하는 경우가 있어, KL-divergence를 이용해 부드럽게 균형맞춰준다.
    + VAE에서 사용한 idea로, $N(0, 1)$과 $\epsilon(t)$의 KL값을 편하게 계산하도록 유도한 공식이 있다. 이걸 사용한다.
$L_{KL} = \sigma^2 + \mu^2 - 1 - \log{\sigma^2}$


### Discovering Edit Directions
+ 위의 Method부분을 보면 좋다.
+ source domain과 target domain의 단어가 포함된 문장을 gpt등의 모델을 통해 생성 (논문에서는 각각 1000개) CLIP기술을 통해 vector로 embedding한 후, 각 단어 벡터의 평균의 차를 구한다.


### Editing via Cross-Attention Guidance
+ 차후 채울 예정..
