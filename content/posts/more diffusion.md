## 참고자료
+ blogs
    + (lil log, what are Diffusion Models?) : https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
    + (Yang song, Generative Modeling by Estimating gradients of the data distribution) : https://yang-song.net/blog/2021/score/
    + 한국 블로그 : https://deepseow.tistory.com/61
    + improved DDPM notion : https://sang-yun-lee.notion.site/Improved-Denoising-Diffusion-Probabilistic-Models-efa847335aef4163bfd3ee96c176f659
    + Diffusion models beats GANs : https://sang-yun-lee.notion.site/Diffusion-Models-Beat-GANs-on-Image-Synthesis-eb1f3826618d42e89d92e489c39f1371
+ papers
    + (Improved Denoising Diffusion Probablistic Models) : https://arxiv.org/pdf/2102.09672.pdf
    + (Diffusion models beats GANs on Image synthesis) : https://arxiv.org/pdf/2105.05233.pdf

+ youtubes
    + (improved DDPM) : https://www.youtube.com/watch?v=8dchQOqvrCE
    + (Diffusion models beats GANs) : https://www.youtube.com/watch?v=bSqA2AIaHy8&t=327s

# Improved DDPM
(contribution)
1. reverse할 때 variance term도 어느정도 학습 하게 해서 NLL(negative log-likelihood)값을 낮추었다.
2. variance값 스캐쥴링을 기존의 linear한 것에서 다른 방법으로 바꾸었다.
3. importance sampling기법을 통해 gradient noise를 줄였다.
4. subsequence를 잡아 sampling speed를 향상시켰다.

### (review) DDPM

Data distribution $x_0 \sim q(x_0)$ 가 주어졌다고 했을 때, q에 대해 forward process를 각 t step에 대해 $\beta_t$를 통해 스캐쥴링한다. 이 때, 마르코프 연쇄를 통해 다음과 같이 정의할 수 있다.
+ $q(x_t|x_{t-1}) = N(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t 𝐈)$

+ 이때, 충분히 큰 T를 통해 $x_T$를 만들면, 결국 가우시안 분포를 따르도록 가정한다.

이제, 이상적인 reverse process(=sampling) $q(x_{t-1}|x_t)$를 직접적으로 계산할 수 없어서 NN($p_{\theta})$을 통해 다음과 같이 근사한다.
+ $p_\theta(x_{t-1}|x_t) := N(x_{t-1};\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$

이렇게 잘 정의한 $p_\theta$를 variational lower bound를 통해 loss를 구해주면, 다음과 같다.  
<img src="https://drive.google.com/uc?id=1RwwODpl9mnBf764naqmR2vxO0ZWGHT-c" height=130>

여기에서 $L_0$와 $L_T$는 DDPM에서 고려하지 않았고, 1 ~ T-1 의 loss를 계산하기 위해 다음과 같이 marginal을 정리하였다.
+ $q(x_t|x_0) = N(x_t;\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$
+ $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$

이제, x_t와 x_0가 주어졌을 때 (reverse에서)예측하고자 하는 posterior $q(x_{t-1}|x_t, x_0)$를 다음과 같이 가우스 커널을 통해 계산한다. 그러면, 원하는 $\tilde{\beta}_t$, $\tilde{\mu}_t(x_t, x_0)$값이 유도된다.

<img src="https://drive.google.com/uc?id=1qMTob1cWcBwvgs01w_xM8mlT6d9rN7Vs" width=700>

위에서 유도한 $\tilde{\mu}$와 $\tilde{\beta}$에서, $\tilde{\beta}$는 forward에서 사용한 $\beta$를 사용하고 $\tilde{\mu}$값만 이용해 대입해서 다음과 같이 구하고자 하는 평균에 대한 값의 파라미터화를 유도한다.
+ $\mu_\theta(x_t, t) = \cfrac{1}{\sqrt{\alpha_t}}(x_t - \cfrac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t))$

이것을 가지고 각 t step에다가 대입해서 앞의 계수 항들 간단히 하여 최종적으로 다음의 단순화 된 loss term을 얻게 된다.
+ $L_{simple} = E_{t, x_0, \epsilon}\big[||\epsilon - \epsilon_\theta(x_t, t)||^2\big]$

앞으로 이 논문에서 주목할 점은 variance이다.

## Improving the log-likelihood


### Learning $\Sigma_\theta(x_t, t)$
이전의 DDPM에서는 variance는 학습하지 않도록 하였다. 심지어, 유도된 posterior를 사용하지도 않고 원래의 forward에서의 variance값을 사용했었다.

+ i.e. $(posterior) : \tilde{\beta}_t = \cfrac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$
+ but, not use $\tilde{\beta}$.
+ $\sigma_t^2 = \beta_t$

이 때, $\beta_t$ : upper bound, $\tilde{\beta_t}$ : lower bound라고 이야기하였다.

그렇다면, 기존 DDPM에서 저 분산값 사용에 있어서 NLL값 손해가 있지 않을까 해서 값 실험을 해보았다.


<img src="https://drive.google.com/uc?id=1m66oJgJHk1aiUtRhvY3o0T-GSE6u_cac" width=450> <img src="https://drive.google.com/uc?id=1H5_b3bGferNzzHAxG6-T2WMIjId69-hW" width=400>

1. time step의 완전 초반부를 제외하면, 두 값이 비슷하므로 t값이 커지게 되면 분산부분은 sampling 퀄리티에 크게 영향을 미치지 않는다
2. 초반의 loss term의 step을 보변 전체 loss에 기여하는 정도가 상당히 크다. 따라서, 기존에처럼 고정된 variance를 사용하기보다, 학습시켜서 likelihood를 확 낮출 수 있을듯하다.

이 때, 우리가 학습하고자 하는 $\Sigma_\theta$는 작은 값이고, NN을통해 직접적으로 예측하는 것은 어렵기 때문에 $\beta$와 $\tilde{\beta}$를 interpolation하고, 그 정도를 학습하도록 하였다.
+ $\Sigma_\theta(x_t, t) = \exp(v\log{\beta_t} + (1 - v)\log{\tilde{\beta}_t})$

__sigma 값을 저 둘 사이의 값으로 가정하는 근거는??__

(2015, Deep Unsupervised Learning using
Nonequilibrium Thermodynamics) : __diffusion 시초 논문__  
<img src="https://drive.google.com/uc?id=1jMJlW65-yyW31DbdKYDzAXKWbNr5wUXy" height=250>  
conditional entropy의 관점에서 보았을 때, 이러한 관계가 성립하고, 가우시안 커널을 이용해 구한 값에 대해서 bounded 값을 잘 설명할 수 있다.

최종적으로, $L_{simple}$은 분산값에 영향을 받지 않으므로 추가 loss term을 추가해준다. 즉, 기존에 학습이 잘 되는 것이 증명이 된 simple텀을 메인으로 두고, vlb텀이 분산값에 따라 guide해주도록 학습한다.
+ $L_{hybrid} = L_{simple} + \lambda L_{vlb}$

$\lambda$는 0.001, vlb term에서 $\mu_\theta$부분은 stop gradient를 적용하여 평균은 simple이 main으로 학습하도록 하였다.

<img src="https://drive.google.com/uc?id=1tERAvVHdWO5W4NOZqgqxrUyUiDWv056-">

200K 4K cosine부분을 살펴보면, simple과 vlb는 사용하면 FID, NLL에 대해 trade off를 가짐을 알 수 있다.

### Improving the Noise Schedule

<img src="https://drive.google.com/uc?id=1y9ODNFhm8VboZ2fCRBTAkVr0-7mUSaSr" height=200>
<img src="https://drive.google.com/uc?id=1uDdvD0dhylXUFaxNRr7WBPjmLO9DkesG" width=400>

기존의 linear한 scaduling은 noise가 너무 초반부터 들어가서 information을 빨리 붕괴시키는 경향이 있다. 따라서 이를 좀 천천히 들어가게끔 cosine을 통해 다시 정의해주었다.
+ $\bar{\alpha}_t = \cfrac{f(t)}{f(0)}$, $f(t) = cos\bigg(\cfrac{t/T + s}{1 + s} \cdot \cfrac{\pi}{2} \bigg)^2$

이를 통해, 필요하다면 역으로 $\beta_t$도 쉽게 계산할 수 있다.

### Reducing Gradient Noise

NLL 관점에서, hybrid loss보단, vlb loss를 그냥 쓰는 것이 당연히 좋지만 실제로 이를 optimize하는 것은 상당히 어려웠다고 한다.

<img src="https://drive.google.com/uc?id=14Hwrh_L0otqOKVfAYzhusjouF-vuKjKh" height=250> <img src="https://drive.google.com/uc?id=1kTkGmgWwLf3Q327Pe6qGIi0HS3WerH_S" height=250>

우선, 가장 큰 문제점으로 본 것은 gradient noise가 vlb의 경우 컸다는 것이다.(가정)  
+ gradient noise ( != value noise) : optimization(대표적으로 SGD)을 하는 중, 전체 데이터 셋과 미니배치 셋으로부터의 gradient의 오차  
<img src="https://drive.google.com/uc?id=1EC3_W0AxCo7OAR5pJZDIPnfG2zAXD1Jw" width=300>   
+ batch size가 클 수록, 당연히 전체 데이터셋의 gradient대로 잘 가고 빠르게 움직인다. 하지만, 반대의 경우 gradient noise가 커지게 된다.

vlb에서의 variance를 낮추는 것에 집중하는데 그러기 위해선 이유를 알아야 한다. 이 논문에선 훈련할 때, vlb의 경우 loss에 영향을 미치는 정도가 time step마다 다른데(figure2), t를 uniformly하게 sampling하기 때문이라고 한다.  
따라서, 중요도에 따른 sampling을 제안한다.
+ $L_{vlb} = E_{t \sim p_t}\bigg[\cfrac{L_t}{p_t} \bigg]$, where $p_t \propto  \sqrt{E[L_t^2]}, \sum{p_t} = 1$


이 때, $E[L_t^2]$ 값은 매번 계산하는 값이기 때문에, 훈련동안 dynamically하게 갱신해준다(앞의 10개 값을 이용).  
+ 각 loss의 값을 일정하게 scaling해주는 대신에, 중요도가 높은 loss는 자주 sampling되어 갱신시켜준다.

### Improving Sampling Speed

본 논문은 모델을 4000번의 diffusion step을 훈련하도록 하였다. 이 때, sampling을 하면서 걸리는 시간과 GPU가 굉장히 크다. 하지만, 이 논문의 모델은 sub sequence를 통해 sampling 하여도 이미지의 퀄리티를 유지할 수 있다.

+ $\beta_{S_t} = 1 - \cfrac{\bar{\alpha}_{S_t}}{\bar{\alpha}_{S_{t-1}}}$, $\tilde\beta_{S_t} = \cfrac{1 - \bar{\alpha}_{S_t}}{1 - \bar{\alpha}_{S_{t-1}}}\beta_{S_t}$
이와 같이 잡게되면, 잘 샘플링 되는데 왜 잘 되는지 생각해보면,

기존의 DDPM은 reverse할 때 $\beta_t$값 만을 사용했 기 때문에, time step을 건너뛰면 그 중간 정보가 소실된다. (DDIM에서의 실험 생각해보면 그 차이가 상당히 큼을 알 수 있다. 즉, $\beta$ 를 사용한것과 $\tilde\beta$를 사용했을 때의 차이가 드러난다.)  
하지만, 본 논문에서는 $\tilde\beta_t$를 사용하였고(정확하게는 interpolation한 값인 $\Sigma_\theta(x_t, t)$), 그렇기 때문에 NLL, FID에서의 score 선방을 할 수 있었다.







# Diffusion Models Beat GANs  on Image Synthesis

## Contributions
1. 모델 아키텍쳐 tuning
    + attention head 증가
    + 다양한 resolution 층에 attention 사용(기존에는 16x16 -> 8x8, 16x16, 32x32)
    + Big GAN의 residual block을 upsampling, downsampling에 사용
    + AdaGN 적용
2. Classifier Guidence 사용하여 FID score 상승 (기존의 GAN 모델을 이김!)

## Back ground

### Improved DDPM


$\Sigma_\theta(x_t, t) = \exp(v\log{\beta_t} + (1 - v)\log{\tilde{\beta}_t})$

### Sample Quality Metrics

#### Inception Score
+ ImageNet에서 한가지 class의 distribution에 대해 고정시켜 학습시키고 그것에 대해 __Sharpness, Diversity__를 계산한다.

    + __모델이 small subset에 대해 적합해져도 점수가 잘나오는 문제점이 존재.__

$S = \exp{(E_{x \sim p}\big[\int{c(y|x)\log{c(y|x)}}dy\big])}$   
$D = \exp{(E_{x \sim p}\big[\int{c(y|x)\log{c(y)}}dy\big])}$  

S : Sharpness
+ classifier가 확신을 가지고 predict하는가
+ S가 증가하면, c(y|x)의 엔트로피가 감소한다. -> 데이터들이 잘 분리되어있다   

D : Diversity
+ 얼마나 다양하게 생성하는가
+ D가 증가하면, marginal distribution인 c(y)의 엔트로피가 증가한다. -> 데이터들이 다양하다  

$IS = S \cdot D$

이 때, classifier인 $c(\cdot)$은 ImageNet을 학습한 Inception V3 모델을 사용. 그래서 Inception score라는 이름이 붙어졌다.

#### Frechet Inception Score (FID score)
+ 학습시킨 ganerated model로부터 데이터들을 모아두고, test할 데이터를 모아서 각각 같은 모델을 이용해 feature extract를 진행한다. 그리고, 각각의 feature의 분포에 대해 거리를 측정한다. (by wasserstein-2 distance)
    + data space로부터 분포간의 거리를 측정하면 너무 흩어져 있기 때문에 이와 같이 하나의 encoder net을 이용해 모아준다.(manifold hypothesis)
    + __Inception score보다 더 Human Judgement에 가깝다는 장점__

<img src="https://drive.google.com/uc?id=19qWj9jTgu0MvherNeA6n-diFJkE1esGk" width=700>

$FID = ||\mu_T - \mu_G||^2_2 + Tr(\Sigma_T + \Sigma_G - 2\sqrt{\Sigma_T\Sigma_G})$

+ 왼쪽부터 각각 Fidelity, Diversity에 해당한다.
+ 즉, 점수가 낮으면 낮을 수록 퀄리티가 좋고 다양성이 풍부한 데이터라고 볼 수 있다. (정확히는 더욱 testing sample space와 유사한 데이터를 만들어낸다.)

## Architecture Improvement

<img src="https://drive.google.com/uc?id=1EfBBN2DThTn8H6TuC5TyHxC07y1zdu0v
" height=250>  
+ 여러가지 tuning 기법을 이용해 모델의 성능을 끌어올렸다 정도..

### Adaptive Group Normalization (AdaGN)

<img src="https://drive.google.com/uc?id=16l0dVbnaZu98gMve_yDwUyCP05801eWS" height=250>


$AdaGN(h, y) = y_sGroupNorm(h) + y_b$

+ Group Normalization을 하는 경우
    1. batch size가 너무 작아 의미 없을 때
    2. NLP의 경우 입력 크기의 다름, 미니배치 분포의 다양성 등등

이 외에도 여러 이유가 있는데, batch norm이 아닌, layer norm을 했을 때 conv net의 대해서도 오히려 학습이 잘되고 성능이 좋은 경우도 있다.   
그래서 요새는 웬만하면 batch norm을 하지만, layer norm (group norm)을 사용하는 경우도 더러 있다고 한다.

__궁금증 : 어떤 상황에서 어떤 normalization을 적용하는 것이 좋은가??__

<img src="https://drive.google.com/uc?id=1k01i58qP0deqrUn0XJocivtholt6oy_W" height=170>  
어쨌거나, time step과 class를 모델의 각 residual에 embedding할 때, 위와 같이 group norm을 할 뿐만 아니라, $y_s, y_b$라를 파라미터를 통해 학습시키며 적용했을 때 FID 점수가 더 잘나왔다고 한다.

## Classifier Guidance
+ 이 논문에서 가장 중요한 부분.

기존의 GAN모델이 FID 점수가 잘 나왔던 이유가, diversity와 fidelity를 trade off로 교환하였기 때문에 퀄리티가 좋았다고 한다.  
즉, Descriminator를 잘 속이기만 하면 되기 때문에 diversity가 낮고 대신에 퀄리티가 좋다.

__DDPM에서도 의도적으로 diversity를 낮춘다면 퀄리티가 올라가고 이를 trade off로서 FID 점수를 높일 수 있지 않을까?__

핵심은 classifier를 어떻게 정의할 것인가이다.(위에 normalizing할 때 class를 정의하긴 했지만 그것 말고 다른 방법을 제시하고 있다.)  
</br>
해당 논문에서는 $p_\phi(y|x_t, t) = p_\phi(y|x_t), \epsilon_\theta(x_t, t) = \epsilon_\theta(x_t)$ 라는 notation을 사용하고 있음을 생각하고 이어지는 수식을 따라가보자.

__목표 : $p_\phi(y|x_t)$를 어떻게 잘 정의할 것인가__  
(이 때, 해당 $p_\phi$는 generating 모델을 y로 유도하게 해주는 역할이다.)

원하는 방향은 다음과 같다.
$p_\phi$를 학습시키게 되면, noise image x_t를 sampling 할 때, gradient $\nabla_{x_t}\log{p_\phi(y|x_t, t)}$를 이용해서 임의의 label y로 유도한다.

### 1. Conditional Reverse Noising Process

기존 diffusion 모델의 unconditional reverse noising process 에서 시작한다.($p_\theta(x_t|x_{t+1})$)

임의의 컨디션 라벨 y 에 대해서 다음의 식이 성립한다.(확률의 성질에 의해)

$p_{\theta, \phi}(x_t|x_{t+1}, y) = Zp_{\theta}(x_t|x_{t+1})p_{\phi}(y|x_t)$(2)   
(Z는 normalizing 상수)  

그러면, 여기로부터 sampling algorithm을 유도할 수 있다.

기존의 diffusion model으로부터 sampling을     
$p_\theta(x_t|x_{t+1}) = N(\mu, \Sigma)$  
라고 간소화 하여 쓸 수 있고,
여기에 log likelihood를 이용해 표현하면 다음과 같다.  
$\log{p_\theta(x_t|x_{t+1})} = -\cfrac{1}{2}(x_t - \mu)^T \Sigma^{-1} (x_t - \mu) + C$ (4)


이 때, $\log{p_\phi(y|x_t)}$가 $\Sigma^{-1}$에 비해 낮은 곡률을 갖는다고 가정한다.  
즉, diffusion step이 무한히 발산하지 않을 때를 가정한다.($||\Sigma|| \rightarrow 0$일 때)  
그러면, $x_t = \mu$일 때 taylor expansion을 이용해 $\log{p_\phi(y|x_t)}$를 근사할 수 있게된다!!!

$\log{p_\phi(y|x_t)}$   
$\simeq \log{p_\phi(y|x_t)}|_{x_t = \mu} + (x_t - \mu)\nabla_{x_t}\log{p_\phi(y|x_t)|_{x_t = \mu}}$ (5)  
$= (x_t - \mu)g + C_1$(6)  
(이 때, $g = \nabla_{x_t}\log_{\phi}(y|x_t)|_{x_t = \mu}$)

이제, (2)의 식에 (4),(6)을 대입해서 전개하고 constant term을 제거해주면 다음의 식 (10)을 얻는다.  
$$\log{(p_\theta(x_t|x_{t+1})p_{\phi}(y|x_t))}\simeq \log{p(z)} + C_4, z \sim N(\mu + \Sigma g, \Sigma)$$    

이제 여기서 log를 양변에서 지워주고나면 남아있는 $C_4$가 처음에 normalization 상수 Z이자, 밑의 gradient scale s를 결정짓는 상수가 된다

이로부터 다음의 sampling 알고리즘이 유도된다!!  
<img src="https://drive.google.com/uc?id=1B2wEue5pywuKq4ZX-nUs0JofhbiPxXLj" height=230>

<img src="https://drive.google.com/uc?id=1K9FN4iL9WUUJRC0JQFLAjG9SQs9lEW-O" width=300>  
DDPM에서의 sampling이 $-\epsilon_\theta$만큼 간다고 생각하면, 거기에 guidance만큼 평행 이동한다고 생각할 수 있다.  
(출처 : https://sang-yun-lee.notion.site/Diffusion-Models-Beat-GANs-on-Image-Synthesis-eb1f3826618d42e89d92e489c39f1371)

전체적으로 sampling되는 그림을 보면 다음과 같이 볼 수 있다.  
<img src="https://drive.google.com/uc?id=1M1zcfn2R0l3BG1HXxv28X9ADYDrzWd5w" height=400>

### 2. Conditional Sampling for DDIM


위의 방식으로 근사하는 것은 확률적 모델인 DDPM에 적용되지만, 결정적 모델인 DDIM에는 사용할 수 없다. (위에서 보여지는 것처럼 z로부터 random성을 이용하기 때문..)

따라서, Song의 논문을 빌려, score-based conditioning trick을 이용해 접근한다.


$\nabla_{x_t}\log{q(x_t)} = -\cfrac{\epsilon_\theta(x_t)}{\sqrt{1-\bar{\alpha}_t}}$ (11)

위의 식은 deterministic sampling에 적용되는 model $\epsilon_\theta(x_t)$에 대해 위와 같은 함수가 유도된다.

$d\mathbf{x} = \bigg[\mathbf{f}(\mathbf{x}, t) - g(t)^2\bigtriangledown_x\log{p_t(\mathbf{x})}\bigg]dt + g(t)d\mathbf{w}$  
를 이용한듯 싶음..

이제, 이를 앞의 절에서 한것 처럼 $p(x_t)p(y|x_t)$에다가 대입해보자.


$\nabla_{x_t}\log{(p_\theta(x_t)p_\phi(y|x_t))} = \nabla_{x_t}\log{p_\theta(x_t)} + \nabla_{x_t}\log{p_\phi(y|x_t)}$  
$= -\cfrac{1}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t) + \nabla_{x_t}\log{p_\phi(y|x_t)}$ (13)

위를 앞의 상수로 다시 묶어서 최종적으로 새로운 noise prediction을 만들어 볼 수 있다.
$\hat{\epsilon}(x_t) := \epsilon_\theta(x_t) - \sqrt{1-\bar{\alpha}_t}\nabla_{x_t}\log{p_{\phi}}(y|x_t)$ (14)

이로부터 아래의 Classifier guided DDIM sampling 알고리즘이 정의된다.  
<img src="https://drive.google.com/uc?id=1sGBhZ3yxq0NpRXZLHoza8lMkVYyKaF6o" height=230>

### 3. Scaling Classifier Gradients


이제, large scale의 generation task에 classifier를 적용하기 위해 먼저 classifier를 ImageNet에 학습을 시켰다고 한다. 이 때의 모델은 UNet 이고, 8x8 size의 layer를 downsampling 시켜 분류 문제를 해결하도록 변형하였다.

그리고, diffusion model에서 사용한 noise와 동일한 분포에 대해서 classifier를 학습시키고, 과적합을 방지하기 위해 randomcrop augmentation을 적용하였다고 한다.

이후, 학습이 종료되고나서 (10)의 식을 이용하여 diffusion모델과 classifier를 결합시켰다.

실험에서 classifier gradient가 (s 값)1이면, 50%정도의 sample 정확도를 가짐을 알 수 있다. 하지만, 실제로 sampling을 해보면 그에 미치지 못한다. 그래서 실험을 하다보니 1보다 큰 값을 잡아야 함을 알게 되었다.

<img src="https://drive.google.com/uc?id=17atQg9cx37yptskZtDWzZl0bXO5KZL34" height=300>

$s \cdot \nabla_x \log{p(y|x)} = \nabla_x \log{\cfrac{1}{Z}p(y|x)^s}$의 관계로부터 이해할 수 있다.  

s값이 커지면, 지수적으로 값이 폭발적으로 오르기 때문데, 더욱 점점 더 sharp 해지게 되고 다시말하면 classifier가 더욱 mode에 집중하게 됨을 의미한다.

__즉, 더 높은 fidelity를 얻고 diversity를 낮추는 것이 된다.__

<img src="https://drive.google.com/uc?id=1-dOLFrhhh7w-OiusSr8nJMQusBibi5h4" height=200>


    classifier gradient값을 높일 수록, FID 점수가 높게 나옴을 알 수 있다.(fidelity 높아짐)
    또한, sFID는 전체 데이터에 대한 값인데, 갈수록 낮게나오는 모습을 통해 모델의 diversity가 낮아짐을 알 수 있다.

<img src="https://drive.google.com/uc?id=1zzw6_enN-aqNSpid3BbijQMrpHG5D3TS" width=600>

정말 모든 데이터 셋에 대해서 기존의 모델을 beat했다

또한 추가적으로, 이미지의 퀄리티도 GAN보다 좋은데 다양성마저 더 좋다.

<img src="https://drive.google.com/uc?id=1XsLyCGPcX-CS7OJ_3-VZGZHnajA0quh5" width=600>
