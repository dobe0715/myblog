# Score Based Model
__(Reference)__
+ 2011(denoising score matching) https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf
+ 2019(score based generative model) https://arxiv.org/pdf/1907.05600.pdf
+ 2021(score based SDE model) https://arxiv.org/pdf/2011.13456.pdf
+ 2021(SDEdit) https://arxiv.org/pdf/2108.01073.pdf
</br>


__(youtube)__
+ (VAE 강의)  https://www.youtube.com/watch?v=o_peo6U7IRM
+ (시립대 강의) https://www.youtube.com/watch?v=HjriJyr8VZ8&list=PLeiav_J6JcY8iFItzNZ_6PMlz9W4_jz5J&index=57
+ (diffusion 강의) https://www.youtube.com/watch?v=uFoGaIVHfoE



## Goal of Generative Model

+ 대상으로 하는 데이터들의 실제 분포 $p(x)$가 존재한다고 가정.
+ 이를 데이터들을 가지고, $p(x)$와 $g_{\theta}(x)$가 유사해지도록 파라미터 $\theta$를 학습시킨다.
+ 이후, 새로운 데이터를 생성한다는 것은 학습한 $g_{\theta}$에 $x_0$를 대입하여  $g_{\theta}(x_0)$ 라는 결과물을 sampling하는 것이다.
+ 이때, $p(x)$의 분포를 모를 때(Implicit), 알 때(Explicit)의 경우 파라미터의 학습방법이 달라진다.

### 1. Implicit : GAN
+ 사실, GAN은 데이터 분포함수 $p, g$를 학습하는 것이 아니고,
+ 생성해낸 데이터 샘플들의 분포, 즉 $p(X), g(X)$가 비슷해지도록 학습하는 것이다.
    + 판별자(discriminator)를 이용해 결과물이 실제 데이터들의 분포에 해당하는지 판가름하여 생성자(generator)를 학습시킨다.



### 2. Explicit
+ Explicit이라고해서 p의 분포를 정확히 아는건아니고.. 결국에 알고있는 분포 (Gaussian 분포)로 간접적으로 근사시키는 방법이다.
+ 진짜로 명백하게 정의하는 모델은 flow모델??
+ 주된 아이디어는, $p_\theta(x)$와 주어진 샘플데이터들$(X_1, X_2, ..., X_n)$을가지고 가장 잘 설명하는 파라미터 $\theta$를 찾기위해 log-likelihood를 계산해, 이 값을 최대화하는 값을 찾는 것이다.



+ __최우추정법(MLE, method of Maximum Likelihood Estimation)__
+ 우도함수 L : likelihood function
+ $f(x|\theta)$ : pdf
+ $x_1, x_2, ..., x_n$ : 관측값
+ $L(\theta|x_1, x_2, ..., x_n) := f(x_1|\theta)f(x_2|\theta). ..f(x_n|\theta)$

<img src="https://drive.google.com/uc?id=1aB4gJ_Mm9h2rC_x1b4LuN9wj9kkRvWG-" height=300>

$\theta^* = \arg \max \limits_{\theta} \sum_{i=1}^n(\log{p_\theta(x_i)})$
요러한 파라미터 $\theta$를 찾는것이 목표
보통은 미분해서 0되는 지점 찾는다.


#### 2.1 VAE, Diffusion
+ __AE(AutoEncoder)__
+ encoder : image -> latent vector ($x$-> $q_\phi(x)$)
+ decoder : latent vector -> image ($p_\theta(q_\phi(x))$)
    + 그런데, 이렇게할 경우 자연스럽게도 학습할때 사용하지 않은 이미지에대한 표현력이 떨어진다(discrete하게 mapping했기 때문에)

+ __VAE(VariationalAutoEncoder)__
+ encoder : image -> latent vector ($x$ -> $q_\phi(z|x)$ -> $z$)
+ decoder : latent vector -> image ($z$ ->$p_\theta(x|z))$ -> $x$)
    + 여기서 q, p는 확률함수(위와 구분지어야함)
+ reparameterization trick : $\epsilon$ ~ $N(0, 1)$, $z = \mu + \sigma^2\epsilon$ -> (continuous하게 mapping해줘서 표현력 좋음)
+ Likelihood : $p_\theta(x) = \int{p_\theta(x|z)p_\theta(z)}dz$ 이걸 최대화 하는 $\theta$찾기!(by ELBO)
    + 여기서 p_theta를 계산할 수 없어서 위의 적분식에 q_phi를 이용해 수식 쭉쭉 전개해나가서 아래와 같은 식을 유도한다.
    + __수식유도 꼭 해보기__
<img src="https://drive.google.com/uc?id=1d4eR5ta0-gOjods4S60vOyR_IUb-oLuD" height="300">

+ __Diffusion__
+ VAE와 상당히 비슷한 느낌. VAE에서는 encoder로부터 나온결과에 노이즈추가했다면
+ 여기선 이전time step과 noise를 interpolation해줌으로써, 적은양의 noise를 time step마다 추가해줘서 가우시안분포까지 확산(diffuse)시킨다, 다음에 그것들을 denoising해준다. 이것을 학습시켜서 가우시안분포로부터 sampling해서 이미지를 생성
+ 마르코프체인을 가정해서 확률함수를 정의하고, 그것으로 likelihood를 최대화 하는 파라미터 계산

<img src="https://drive.google.com/uc?id=19lhFdBzXRkc7Z9_0wOffcPXxnfJk0ZL8">

__생성모델들 그림__
<img src="https://drive.google.com/uc?id=19iPqI3MYLI9fg7bIAsyEwTIn9HUsCKSL" height="500">

# Score based Model

#### Energe Based Model(EBM)
+ 어떤 x로부터의 분포 y가 있다고 하자. ex) $y=x^2$
+ 이를 파라미터를 통해 표현하면, $E_\theta(x)=x^2$과 같이 볼 수 있다.
+ 이를 확률함수로 만들어준다. $p_\theta(x)=\cfrac{e^{-E_\theta(x)}}{z_\theta}$, $\int{p_\theta(x)}dx = 1$

이제, 이 p에대해 likelihood를 최대화한다!!

#### Score Matching
+ __Score__ : $\nabla{x}\log{p(x)}$값.(데이터들의 확률함수의 log-likelihood의 gradient값)
+ traditional한 방법 : $\arg\min\limits_{\theta}\mathbb{E}_{p(x)}\frac{1}{2}[||\nabla{x}
\log{p(x)} - S_\theta(x)||_2^2]$

<img src="https://drive.google.com/uc?id=1eSXNTlAvzu7y1rcli8B65MS-QMJoOihv" height="200">  
그런데, p의 분포를 모르는 상황이기때문에  
+ $\arg\min\limits_{\theta}\mathbb{E}_{p(x)}\frac{1}{2}[||S_\theta(x)||_2^2 + tr(\nabla{x}S_\theta(x)]$를 구한다. 그런데, 이때 이 gradient의 대각합을 구하는 과정에서 각 x마다 기울기값을 계산해야하는 불상사가 발생한다. (ex) 100,000개의 데이터가지고 학습하면 100,000차원의 기울기계산해야함


#### Denoising Score Matching
+ 기본 idea : 알고있는 분포 q를 이용해(보통 Gaussian 분포) x에 Noise($\sigma$)를 추가한 데이터의 분포를 $q_\sigma(\tilde{x}, x)$라 정의하고, 이놈을 이용해 score를 정의한다.

$$q_\sigma(\tilde{x}, x) = q_\sigma(\tilde{x}|x)q_0(x)$$
+ $q_\sigma(\tilde{x}, x)$ : noise가 추가된 데이터 분포
+ $q_\sigma(\tilde{x}|x)$ : noise 분포
+ $q_0(x)$ : 원래 분포

$q_\sigma(\tilde{x}) = \int{q_\sigma(\tilde{x}, x)}dx = \int{q_\sigma(\tilde{x}|x)q_0(x)}dx \simeq \int{q_\sigma(\tilde{x}|x)p_{data}(x)}dx$  
이때, $\sigma$가 충분히 작으면 위와같이 근사할 수 있다고함.
(당연히도 noise가 충분히 작다면 원본이미지와 그게 다르지 않을것이라.. diffusion 모델에서 작은 noise추가할 때 가정한 것과 같은원리)

이제, 목적함수를 정의하자.
$\sigma$를 크기순으로 나열해서,
+ $\sigma_{min}=\sigma_1 < \sigma_2 < ... < \sigma_{N}=\sigma_{max}$
+ $p_{data} \simeq p_{\sigma_{min}}, p_{\sigma_i}(\tilde{x}|x) = N(\tilde{x};x, \sigma_i^2)$으로부터
+ $\theta^* = \arg\min\limits_{\theta} \sum_{i=1}^N{\sigma^2 \mathbb{E}_{p_{data}}\mathbb{E}_{p_{\sigma_i}(\tilde{x}, x)}[||S_\theta(\tilde{x},\sigma_i)-\nabla_{\tilde{x}}\log{p_i}(\tilde{x}|x)||_2^2]}$를 계산한다.
이 때, $-\nabla{\tilde{x}}\log{p_{\sigma_i}}(\tilde{x}|x) \simeq \cfrac{\tilde{x}-x}{\sigma_{i}^2}$ : Gaussian Kernel로 근사하여 계산할 수 있다.

#### Sampling with Langevin Dynamics
+ Langevin Dynamics : 주어진 데이터 x의 score를 알고 있을 때,
+ $z_t\sim N(0, 1), \epsilon$을 이용해,
+ $x_t = x_{t-1} + \frac{\epsilon}{2}\nabla_x\log{p(x_{t-1})} + \sqrt{\epsilon}z_t$의 연산을 반복수행한다.
이 때, 잘 학습되었다는 가정하에
+ $x_t = x_{t-1} + \frac{\epsilon}{2}s_\theta(x_{t-1}) + \sqrt{\epsilon}z_t$을 통해 sampling을 해준다!!

<img src="https://drive.google.com/uc?id=15Oe7uhjLH8FcxuaH0Rc9FGMZyrP2fIsJ" height="200">

__NOTE__: Markov Chain Monte Carlo, MCMC
+ Monte Carlo : 통계를 통해 시뮬레이션 수행하여 원하는 값을 얻는 기법
    + 원주율 계산 : 찍는 점의 개수를 늘려가며 원주율 근사한다.
    + <img src="https://drive.google.com/uc?id=1axbZPD2S4ZrEx67eNpxivq5qqM6h5-4K">
+ Markov Chain : 현재 state가 바로 직전 state의 영향만 받는 확률 과정
$$p(x_t|x_{t-1}, x_{t-2}, ..., x_0) = p(x_t|x_{t_1})$$



# Score based Generative modeling through SDEs
+ (2021) score를 SDE를 통해 접근한다.


## Recall to DDPM
<img src="https://drive.google.com/uc?id=19lhFdBzXRkc7Z9_0wOffcPXxnfJk0ZL8">

### Forward Process
+ 원본이미지 $x_0$에 미세한 Gaussian noise를 hierarchical하게 추가해주는 과정이다.

각 time step에 대해서는
+ $q(x_t|x_{t-1}) = N(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t 𝐈)$
    + $x_{t-1}$에 $\sqrt{1-\beta}$만큼 곱해놓고 $\beta_t$ 만큼의 분산을 통해 sampling하는 것이 곧, $x_t$에 noise를 추가한 것이 된다.
    + $\beta$값이 커질 수록, noise가 커진다고 이해할 수 있다.(여기선, 0.0001~0.02로 고정함)

이 때, 마르코프 연쇄에 의해
+ $q(x_{1:T}|x_0) = \Pi_{t=1}^{T} q(x_t|x_{t-1})$
를 만족한다.

### Reverse Process
+ Noise가 추가되어있는 $x_T$로부터 denoising을 해주는 과정이다.

각 time step에 대해
전통 diffusion
+ $p_\theta(x_{t-1}|x_t) = N(x_{t-1};\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$
    + 각 time t에대한 평균과 분산을 예측하게만드는 파라미터 $\theta$를 학습해야한다.

DDPM reverse process
+ $\Sigma_\theta(x_t, t) = \sigma_t^2I$
+ $\sigma_t^2 = \tilde{\beta_t} = \cfrac{1-\tilde{\alpha}_{t-1}}{1-\tilde{\alpha}_t} \beta_t$ or $\sigma_t^2 = \beta_t$
    + ($\beta_t$ 에 의존하기 때문에 변경)
+ $\mu_\theta(x_t, t) = \cfrac{1}{\sqrt{\bar\alpha_t}}\bigg(x_t - \cfrac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\bigg)$
    + 평균값$\mu$을 구하는 것이아니라, 잔차$\epsilon$를 구하자! -> U-Net 사용
    + 잔차를 통해 $\mu$를 구할 수 있고, $x_{t-1}$을 샘플링할 수 있다.


마르코프 연쇄에 의해,
+ $p_\theta(x_{0:T}) = p(x_T)\Pi_{t=1}^T p_\theta(x_{t-1}|x_t)$

### Loss Function
전통 diffusion
+ $\mathbb{E}_q\bigg[D_{KL}(q(x_T|x_0) || p(x_T)) + \sum_{t>1}{D_{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t)) - \log(p_\theta(x_0|x_1))}\bigg]$

DDPM
+ $\mathbb{E}_q\bigg[\sum_{t>1}{D_{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t)) - \log(p_\theta(x_0|x_1))}\bigg]$
    + regularization term을 제거하고, 위에서 정의한 관계식으로 정리하면,

다음과같이 단순화 할 수 있다.
+ $\mathbb{E}_{t, x_0, \epsilon}\bigg[||\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, t)||^2\bigg]$, $\epsilon \sim \mathcal{N}(0, I)$

__수식 유도 차근차근 스스로 해보기!__

어쨌든, 위의 수식으로부터, $\epsilon_\theta = s_\theta$라고 보면,

sampling과정을 다음과같이 표현할 수 있다.
+ $x_{i-1} = \cfrac{1}{\sqrt{1-\beta_i}}(x_i + \beta_i s_\theta * (x_i, i)) + \sqrt{\beta_i}z_i$, $i = N, N-1, ..., 1$

## SDE 관점에서 해석
+ 주가예측 연속시간모형에도 사용된 방정식이라고함..
(Ornstein-Uhlenbeck process) https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

$$d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)dt + g(t)d\mathbf{w}$$
+ $t$ : continuous time, $\{x(t)\}_{t=0}^T$ : diffusion process간 x값
+ $x(0) \sim p_0$, $x(T) \sim p_T$ : i.i.d samples


+ $\mathbf{w}$ : Standard Wiener process(_Brownian motion_)
    + t에 대한 noise term, 위의 $\sigma$역할
    + t에 따라, $w_{t+\vartriangle{t}} - w_t : \Omega \rightarrow \mathbb{R}^d$, $w_{t+\vartriangle{t}} \sim \mathcal{N}(0, \vartriangle{t})$

+ $f(\cdot, t) : \mathbb{R}^d \rightarrow \mathbb{R}^d$ : x 에 대한 drift coefficient function
    + 시간에 대한 추세(경향)을 반영햔 term

+ $g(\cdot) : \mathbb{R} \rightarrow \mathbb{R}$ : $x(t)$에 대한 diffusion coefficient function
    + 이 때, g는 x와 독립이어야 한다.

어쨌거나, 위와같이 표현된 식은 다음의 식으로 reverse가 가능하다고한다.
$$d\mathbf{x} = \bigg[\mathbf{f}(\mathbf{x}, t) - g(t)^2\bigtriangledown_x\log{p_t(\mathbf{x})}\bigg]dt + g(t)d\mathbf{w}$$
$0<s<t<T$, $p_{st}(x(t)|x(s))$

### SMLD(score matching langevin dynamics)

#### Forward process
$p_{\sigma_i}(\tilde{x}|x) = N(\tilde{x};x, \sigma_i^2)$로부터,
$x_i = x_{i-1} + \sqrt{\sigma_i^2 - \sigma_{i-1}^2}z_{i-1}$를 끌어올 수 있고, 이로부터 미분방정식을 유도하면,(오일러 메소드)

$dx = \sqrt{\cfrac{d[\sigma^2(t)]}{dt}}dw$

이는 위의 구조와 동일하다.
</br></br></br>
#### reverse process
수식유도가.. 다음에 해봐야할듯

### DDPM
#### Forward process
$q(x_t|x_{t-1}) = N(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$로부터,
$x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta}z_{t-1}$를 끌어올 수 있고, 이로부터 미분방정식을 유도하면,(오일러 메소드)

$dx = -\cfrac{1}{2}\beta(t)xdt + \sqrt{\beta(t)}dw$
</br></br></br>

#### Reverse process
이것도 유도가... 다음에
