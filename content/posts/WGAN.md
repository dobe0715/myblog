# Review : Wasserstein GAN
+ 새로운 metric을 통한 GAN의 loss를 제시.
+ 분류모델같은경우는 정답이 확실히 정해져있어서 모델의 loss를 확실히 정량화 할 수 있었다.(cross-entropy, mse 등)
+ 하지만 생성모델은 어느정도가 정답과 가까운지 정량적으로 측정하기 힘들다. 이러한 것을 해결하기 위한 metric이다.

## 4가지 거리개념
+ 주어진 가정
1. $\mathcal{X}$ : compact metric(유계인 닫힌 공간)
2. $\Sigma$ : $\mathcal{X}$의 Borel subset들의 집합 (측정가능한 집합에 대해서만 확률을 논하겠다)
3. $Prob(\mathcal{X})$ : $\mathcal{X}$의 확률측도공간

두 분포 $\mathbb{P}_r, \mathbb{P}_g \in Prob(\mathcal{X})$에 대하여...

### 1. TV(Total Variation) distance
$$\delta(\mathbb{P}_r, \mathbb{P}_g) = \sup_{A \in \Sigma}|\mathbb{P}_r(A)- \mathbb{P}_g(A)|$$
+ 두 분포간에 공통되는 영역에서의 확률값들의 차이에서 가장 큰값
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/79/Total_variation_distance.svg/1200px-Total_variation_distance.svg.png" height=400>

### 2. KL(Kullback-Leibler) divergence
$$KL(\mathbb{P}_r| \mathbb{P}_g) = \int{\log{(\cfrac{P_r(x)}{P_g(x)})P_r(x)}}d\mu(x)$$
+ 두 분포간 엔트로피정보의 차이
+ 비대칭성(거리개념 x)

### 3. JS(Jensen-Shanon) divergence
$$JS(\mathbb{P}_r| \mathbb{P}_g) = KL(\mathbb{P}_r| \mathbb{P}_m) + KL(\mathbb{P}_g| \mathbb{P}_m)$$  
$$\mathbb{P}_m = (\mathbb{P}_r+ \mathbb{P}_g)/2$$

+ KL의 단점인 비대칭성 보완.
+ 거리개념으로 사용가능

#### 측정값 살펴보기
+ 두 평행선 분포를 비교.
+ $Z \sim U[0, 1]$ 
+ $\mathbb{P}_0 : \mathbb{P}_0 \sim (0, Z) \in \mathbb{R}^2$ 
+ $\mathbb{P}_\theta : g의 분포, g_\theta(z)=(\theta, z)$  

<img src="https://drive.google.com/uc?id=1PlPjt4g9kn8MFhpOcIvNw6CuYfBSiexV" width=350>



$\delta(\mathbb{P}_0, \mathbb{P}_\theta) = \begin{cases} 1 & \text{}\; \theta \neq 0 \\ 0 &\text{} \; \theta = 0 \end{cases}$  
$KL(\mathbb{P}_0| \mathbb{P}_\theta) = \begin{cases} +\infty & \text{}\; \theta \neq 0 \\ 0 &\text{} \; \theta = 0 \end{cases}$  
$JS(\mathbb{P}_0| \mathbb{P}_\theta) = \begin{cases} \log{2} & \text{}\; \theta \neq 0 \\ 0 &\text{} \; \theta = 0 \end{cases}$

+ __두 분포가 겹치는 부분이 없으면, 상수값이거나 발산하게 된다. 즉, 거리계산을 못한다.__

### 4. EM(Earth-Mover) distance (= Wasserstein-1 distance)
$$W(\mathbb{P}_r, \mathbb{P}_g) = \inf_{\gamma \in \prod(\mathbb{P}_r, \mathbb{P}_g)} \mathbb{E_{(x, y) \sim \gamma}}[|x - y|]$$  
$\gamma \in \prod(\mathbb{P}_r, \mathbb{P}_g)$ : $\mathbb{P}_r, \mathbb{P}_g$ 분포 사이의 결합확률함수
+ $\gamma(x, y)$ : $\mathbb{P}_r$ 에서 $\mathbb{P}_g$ 분포로 옮기게 될 질량(mass)
+ $\inf{\mathbb{E}}$ : 질량을 옮기는 것에 있어 드는 "최소비용"

#### Example

EX 1)  
<img src="https://drive.google.com/uc?id=1VTk84xOJMcPFnpGWPMydseg5sGJMX0Eo" height = 500>  

EX2) : 최소의 상황  
<img src="https://drive.google.com/uc?id=1wmpXM_DVvovA5Ouotk1LYc3uXC4Cc0Js" height=300>  
* (수정) $\gamma(1,1) = \cfrac{1}{3}$

__이러한 성질 때문에 earth mover라는 이름이 붙여졌다__

#### 측정값 살펴보기
<img src="https://drive.google.com/uc?id=1PlPjt4g9kn8MFhpOcIvNw6CuYfBSiexV" width=350>  
+ $W(\mathbb{P_{\theta}}, \mathbb{P}_0) = |\theta|$
+ __두 분포가 겹치지 않아도 거리를 측정할 수 있다!!!__



## Warsserstein distence 구하기
+ $f : \mathcal{X} \rightarrow \mathbb{R}$, 1-Lipschitz 함수(기울기가 1을 넘지 않음) 일때, 

$$W(\mathbb{P_{r}}, \mathbb{P}_{\theta}) = \sup_{|f|_L < 1}(\mathbb{E}_{x\sim\mathbb{P}_r}[f(x)]-\mathbb{E}_{x\sim\mathbb{P}_\theta}[f(x)])$$

라는 정리가 있다. 이것을 본 논문에서는 목적함수로 설정하였다.


#### 수식 해석
우리의 목적은 __Generator가 실제 이미지(도메인)와 비슷한 이미지를 만들어 내는 것__이다.  
즉, $g : Z \times \mathbb{R}^d \rightarrow \mathcal{X}$인 함수를 구성해서, 
latent space의 벡터 z를 실제 이미지 x로 mapping해주는 함수로 학습시키는 것이다.

그리고 그러한 g의 파라미터$\theta$에 대해 $g_{\theta}(z)$의 분포가 $\mathbb{P}_\theta$이다.

이제, 실제 데이터 분포인 $\mathbb{P}_r$와$\mathbb{P}_\theta$의 간격을 줄이는 것을 목적이고 그러한 함수로 위와 같이 설정할 수 있다.

$f_w$ : critic(discriminator)  
$g_\theta$ : generator

$$\max_{w \in \mathcal{W}}(\mathbb{E}_{x\sim\mathbb{P}_r}[f_w(x)]-\mathbb{E}_{z \sim p(z)}[f_w(g_{\theta}(z))])$$

이때, f가 립쉬츠 조건을 만족하기 위해서 w를 가중치클리핑을 통해 제한했다.  
즉, 치역을 강제로 작게 만들어서 함수의 기울기가 작아지게 만들음..  
-> __끔찍한 방법이라고 소개하고있고, 학습시간이 오래걸리는 원인이 된다.__  

차후에 가중치 클리핑이아니라 기울기 페널티(GP, Gradient Penalty) 항을 추가하여 이 문제를 해결하였고, 그러한 WGAN-GP가 다른 GAN모델에 많이 사용된다.


## Wasserstein distance와 Mode collapse

<img src="https://drive.google.com/uc?id=1AkosiSyzzFnbgYaBwD5w_2I-Q9eLehAP" width=500>  
+ 기존 GAN의 JS거리를 사용하면, 만약 fake 이미지 분포가 실제 이미지와 멀리 떨어져있게되면, distriminator가 학습을 부분적으로 하게 되고 최적화되지가 않는다.
    + 기울기 vanishing이 생기고, 더이상 가중치 갱신이 되지 않는 부분이 생긴다.
    + generator가 그부분만 계속 학습해버린다면.. 모드붕괴가 발생

+ WGAN을 사용했을 때에는 모든 분포에 대해 학습을 잘 할 수 있기 때문에 위와 같은 상황이 발생하지 않는다.

## Wasserstein distance를 활용한 모델평가지표
+ 두 분포간의 거리를 연속적으로 정량화 할 수 있다는 점에서 아주 유용하다.


### MS-SSIM
+ 다중 스케일 구조적 유사도(multi-scale structural similarity)
+ pg gan 모델에서 제시한 방법으로, 결과물을 다양한 scale에서 분석할 수 있다.
    + SWD(Sliced Warsserstein Distance)라는 방법을 이용.
    + 기존의 WD의 계산은 차원이 높을 경우 시간복잡도가 커진다. 이것을 1차원으로 정사영시켜서 빠르게 계산해주는 방법.

__방법__  
1. 원하는 양만큼 image를 (실제, 생성 각각)sampling한다.
2. Laplacian pyramid를 만들어 descriptor(이미지를 설명하는 특징벡터)를 추출한다.
    + Laplacian pyramid란, 같은 이미지로부터 upsampling, downsampling을 하면서 계층적으로 해상도를 피라미드 형태로 만들어주는 것.
3. 추출한 descriptor에 대해 color channel마다 normalization
4. 실제에서 추출한 것과 생성에서 추출한 것에 대해 SWD를 계산한다.

### FID
+ Frechet Insception Distance
    + 실제, 생성이미지를 셈플링한다.
    + 사전훈련된 Inception V3에 적용하여 특징벡터 분포를 추출한다. 
    + 결과를 frechet distance(Wasserstein-2 distance)로 계산한다.
