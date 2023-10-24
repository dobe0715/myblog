# Variational Diffusion Models(2021)
참고링크
+ 유튜브 : https://www.youtube.com/watch?v=yR81b3UxgaI&t=1354s
+ 노션 : https://sang-yun-lee.notion.site/Variational-Diffusion-Models-f72d9cb1a2004a9088470c95cdc929e3
+ 논문 : https://arxiv.org/abs/2107.00630

## Contributions
1. likelihood SOTA 찍었다.
2. 모델 파라미터($w$)와 동시에 noise schadule($\gamma$)도 학습시켰다.
3. VLB를 SNR를 통해 간단히 표현하였다.
4. continuous-time VLB값은 noise schedule과 관련없음을 보였다.(각 끝점과 관련있음)

## Model

### 1. Forward time diffusion process

<img src="https://drive.google.com/uc?id=1uoRuOdVWWVXb5YmkLnzQ_NcHKNLxZwrV" height=350>

+ $x$ : 주어진 데이터
+ $p(x)$ : 측정된 x의 주변분포
+ $z_t$ : $t$ 시간에서의 latent variable

#### 0. definitions  
+ $q(z_t|x) = N(\alpha_t x, \sigma_t^2 I)$  
+ $SNR(t) = \cfrac{\alpha_t^2}{\sigma_t^2}$
    + 이 때, SNR은 단조 감소를 만족시켜야 한다.(차후에 만족하도록 noise 스케쥴함)

__위의 가정에서 시작해서 diffusion process를 모두 설명한다.__

#### 1. forward process  
+ $\forall{t>s}, q(z_t|z_s) \sim N(\alpha_{t|s}z_s, \sigma_{t|s}^2 I)$
    + $\alpha_{t|s} := \cfrac{\alpha_t}{\alpha_s}$
    + $\sigma_{t|s}^2 := \sigma_t^2 - \alpha_{t|s}^2 \sigma_s^2$

##### __잘못된 유도과정__  

<img src="https://drive.google.com/uc?id=1R1t7jAJKInJZKMvdFFv60wfxvxwQhu-6" height=300>

##### __올바른 유도과정__(내 생각에)

<img src="https://drive.google.com/uc?id=12-dso4GYLTCGct85jK8THLwoj2Swpdqa" width=550>

#### 2. Markovian process : 가장 최근 것만 본다
+ $\forall 0 \leq s < t < u \leq 1, q(z_u|z_t, z_s) = q(z_u|z_y)$


#### 3. backward process
+ $\forall 0\leq s<t \leq1, q(z_s|z_t, x) = N(\mu_Q(z_t, x;s, t), \sigma_Q^2(s, t)I)$
    + $\sigma_Q^2(s, t) = \sigma_{t|s}^2 \cfrac{\sigma_s^2}{\sigma_t^2}$
    + $\mu_Q(z_t, x;s, t) = \cfrac{\alpha_{t|s}\sigma_s^2}{\sigma_t^2}z_t + \cfrac{\alpha_s \sigma_{{t|s}}^2}{\sigma_t^2}x$

##### __유도과정__  

<img src="https://drive.google.com/uc?id=18V5BcWqpfj50sSRbLTPfpUV7kkF8BYUr" height=300>

<img src="https://drive.google.com/uc?id=1tUUxtRUONhJ3XyxZUeB2zVHYQup2V9mc" width=500>

### 2. Noise schedule

$\sigma_t^2 = sigmoid(\gamma_\eta(t))$
으로 정의하자. ($\gamma$는 단조함수로 차후에 어떻게 만드는지 설명해줌.)  

또한, 우리의 모델은 VP(variance preservation) diffusion process이므로 mean variance는 다음의 관계를 만족한다.  
$\alpha_t = \sqrt{1 - \sigma_t^2}$  
따라서,
+ $\alpha_t^2 = sigmoid(-\gamma_\eta(t))$
+ $SNR(t) = \exp(-\gamma_ega(t))$


### 3. Reverse time generative model
앞서 우리는 backward process도 정의를 했다.
이 때, $T < \infty$이면 discrete time,
$T \rightarrow \infty$이면 continuous time 모델이라고 부른다.

+ $\tau = \cfrac{1}{T}$, $s(i) = \cfrac{i-1}{T}$, $t(i) = \cfrac{i}{T}$
라고 정의하자. 주어진 모델을 해당 표현을 빌려서 쓰면 likelihood는 다음과 같다.  
$p(x) = \int_z{p(z_1)p(x|z_0)\prod_{i=1}^T{p(z_{s(i)}|z_{t(i)})}}$


앞에서부터, 각각의 p(학습할 모델) 함수에 대하여 q(이상적인 모델; unknown)로 close 시키는 것을 목표로 한다.

1. $p(z_1)$  

<img src="https://drive.google.com/uc?id=1eA0wpgPjvPkxTqah5TI-0dK6xvC1dBo2" height=130>


2. $p(x|z_0)$  

<img src="https://drive.google.com/uc?id=1OrERWR0jTsCv6ngIJTG0mgd5j7OWOKdi" height=500>

3. $p(z_{s(i)}|z_{t(i)})$  

<img src="https://drive.google.com/uc?id=1nqBtIRjqsjccwjESc5p9QGkGYSSF0cyg" height=500>

__이 때, 평균을 파라미터화 한 것에 대한 해석이 3가지 존재한다__

1. $x$를 손상시켜서 $z_t$를 만들고 다시 $\hat{x_\theta}$으로 복구시킨다 : __denoising model__
2. 각 time step t 마다 $z_t$로부터 noise $\epsilon$을 직접적으로 추론한다. : __noise prediction model__
3. $q$의 주변밀도함수의 score(i.e score of the marginal density : $s^*(z_t;t) = \triangledown{\log{q(z_t)}}$) : __score model__

모두 같은 목적의 class를 학습한다. 즉, 동치관계를 가진다.  
논문에서는 1번의 해석이 직관적으로 잘 와닿아서 해당 표현을 사용한다.  
다만, 학습할 때는 2번을 사용하였다.(DDPM의 방식)


### 4. Noise prediction model & Fourier features

Noise prediction 모델로 학습하기 전에 input에 각 픽셀에 대해 fourier feature kernel을 사용해 additional channel을 구성하여 모델에 집어넣었더니, likelihood가 좋아졌다고한다.  


추가적으로, 기존의 diffusion모델은 coarse scale의 패턴에 집중한 이미지들이었는데, 이 방식을 이용하여 더욱 fine scale의 디테일한 부분도 잡아낼 수 있었다고 한다.

(이 fourier feature부분 차후에 공부해서 채워넣을 예정)

### 5. Variational lower bound

VLB텀은 잘 알다시피 다음과 같이 유도된다.  
<img src="https://drive.google.com/uc?id=1YQfovyVQ0RFTFn7BbrQycjnpD6-BLnaG" height=180>

## Time Models
그래서, 너네 mean variance 재정의해서 기존의 diffusion모델들 깔끔하게 잘 설명해서 좋아.  
이제 실제로 학습은 어떻게 할껀데??  
라는 질문에 대한 답변들이다.

### 1. Discrete-time model

학습할 모델의 최종 결과는 다음과 같다.  
$L_T(x) = \cfrac{T}{2}\mathbb{E}_{\epsilon \sim N(0, I), i \sim U\{1, T\}}\big[(SNR(s) - SNR(t)) ||x - \hat{x}_\theta (z_t;t)||_2^2 \big]$

그리고, 앞서 정의한 $\gamma_\eta$를 이용하면
$L_T(x) = \cfrac{T}{2}\mathbb{E}_{\epsilon \sim N(0, I), i \sim U\{1, T\}}\big[\exp(\gamma_\eta(t) - \gamma_\eta(s) - 1)||\epsilon - \hat{\epsilon}_\theta(z_t;t)||_2^2 \big]$

해당 식 이용해서 Monte Carlo estimator를 통해 VLB를 maximizing하면서 $\eta, \theta$를 __jointly__ optimize 한다고 한다.

__수식 유도__  

<img src="https://drive.google.com/uc?id=1Cm7cYv7k5DBXSwPxIQssaoCFycQtXJNg" width=600>  
<img src="https://drive.google.com/uc?id=1xBaCChvvrFiKWMkscp0rU92SSYjOxVQm" width=600>

추가적으로, T 값을 2T로 늘려서 진행해봤는데, VLB값이 낮아졌다고 한다.  
(diffusion loss term을 리만합 한다고 생각해보면 직관적으로 $L_{2T}(x) < L_T(x)$임을 알 수 있다.)

<img src="https://drive.google.com/uc?id=1CK7bKM0NiaK0QJlCkb7TQk67MOnDo70S
" heigth=200>

### 2. Continuous-time model

discrete model로부터 $T \rightarrow \infty$이게 해주면 유도할 수 있다.  
최종 식은 다음과 같다.  
<img src="https://drive.google.com/uc?id=1-NXFT28apFCn1P23VdmsXVv0vVmomwln" height=120>  
학습할 때는 마찬가지로 noise prediction 관점으로 보면  
$L_{\infty}(x) = \cfrac{1}{2}\mathbb{E}_{\epsilon \sim N(0, I), t \sim \mathcal{U}(0, 1)}\bigg[\gamma'_\eta(t) ||\epsilon - \hat{\epsilon}_\theta(z_t;t)||_2^2 \bigg]$

__수식 유도__  
<img src="https://drive.google.com/uc?id=1aImktBInmItDKfvU8zGRU_w9qrTihEJX" width=700>

#### continuous diffusion model 의 동치관계로부터 얻을 수 있는 의미



위에서 식 15를 치환적분해주면 다음과 같아진다.  
$L_\infty(x) = \cfrac{1}{2}\mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}\int_{SNR_{min}}^{SNR_{max}}||x - \tilde{x}_\theta(z_v, v)||_2^2$

__NOTE__   
$v=SNR(t)$라 두면,   
SNR은 단조 감소라는 가정이 있었으므로 $t = SNR^{-1}(v)$로부터,   
$\tilde{x}_\theta(z, v) = \hat{x}_\theta(z, SNR^{-1}(v))$ 라 할 수 있고,   
$SNR_{min} = SNR(1), SNR_{max} = SNR(0)$이라하면 치환이 가능해진다.

또한, 서로 다른 noise scadule에 학습한 별개의 모델 에 대해서도 SNR min, max가 같으면 둘이 equal한 관계를 가짐을 보였다.

__결론 : continuous 모델로 SNR min, max만 잘 구성해놓으면 variance를 어떻게 구성해놓아도 다 똑같으니 학습만 잘 되게끔 만들면 된다!__

## Noise schedule : $\gamma_\eta(t)$

앞에서 SNR이 단조감소를 만족시켜야 한다고 했다. 이를 위해서 $\gamma_\eta$를 어떻게 파라미터화 해야 할까? 논문은 다음의 네트워크를 제안한다.

<img src="https://drive.google.com/uc?id=1p23E2IhR02XdE0KNkDNCq6LPkyjvFX4N" heigth=350>

<img src="https://drive.google.com/uc?id=1NeSt-paw-ZNAhs4ktGh7GA89CEhu60VI" width=650>

<img src="https://drive.google.com/uc?id=1J7NEh2a4oB0a4DEGB-OGoeKBK_ByqJb7" width=650>

## Experiments

+ likelihood sota 달성
+ t 늘어날 수록 좋았음. 특히 continuous할 때 가장 좋았다.

<img src="https://drive.google.com/uc?id=1zwpQCXAa6zbetbbtv-DPtjjrb5mnBLWq" width=700> <img src="https://drive.google.com/uc?id=1zMiDjw8Pcsbqm6Ho9zHPh9jId0c9Qstl" width=350>




```python

```
