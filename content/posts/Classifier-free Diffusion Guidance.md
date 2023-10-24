# Classifier-free Diffusion Guidance(2022)

(참고링크)
+ youtube : https://www.youtube.com/watch?v=Q_o0SpXv9kU
+ blog : https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/cfdg/
+ paper : https://arxiv.org/abs/2207.12598

## Contributions
1. unconditional model을 학습하면서 동시에 conditional model을 학습시킬 수 있다.
2. 기존에 비해 학습 파이프라인을 간단화 하였다,

## BackGround

### Diffusion Models Beat GANs on Image Synthesis(Classifier Guidance 제안)

핵심은, conditional 모델의 likelihood 식을 잘 전개하면, unconditional 모델과 classifier 모델로 나눠 학습시킬 수 있다는 것이다.

$p_{\theta, \phi}(x_t|x_{t+1}, y) \simeq Zp_{\theta}(x_t|x_{t+1})p_{\phi}(y|x_t) \simeq \log{p_\theta(z)} + C_4, z \sim N(\mu_\phi + \Sigma_\phi g, \Sigma_\phi)$

이를 노이즈 예측 모델로 대응시키면 다음과 같다.  
$\tilde{\epsilon}(z_\lambda, c) = \epsilon_\theta(z_\lambda, c) - w \sigma_\lambda \nabla_{x_t}\log{p_{\phi}}(c|z_\lambda)$  
</br>
$c$ : condition  
$z_\lambda$ : noised 이미지    
$w$ : classifier guidance weight    

w값을 키우면, 모델의 다양성은 줄어들지만, inception score는 증가한다.

__문제점__
1. 두개의 모델을 적절하게 학습시켜야 한다는 번거로움
2. diffusion sampling을 하는동안에 adversarial attack을 준다고 이해할 수 있다.(온전한 sampling을 못하게 됨.)
    + GAN에 대응 시켜볼 수 있다.
    + generator -> diffusion model
    + discriminator -> classifier

이럴꺼면 GAN을 잘 발전시키지, 굳이 diffusion을 사용할 이유가 없어진다.

## Classifier-Free Guidance

위에서 유도한 식으로부터,논문의 notation으로 바꿔서 써보면   
$\tilde{p_\theta(z_\lambda | c)} \propto p_\theta(z_\lambda|c)p_\theta(c|z_\lambda)^w$  
이고,

여기서, 비례관계이기 때문에 w값을 w+1로 바꿔줄 수 있다. 그러면, 이렇게 바꾼 값을 score에서 살펴보면

$\epsilon_\theta(z_\lambda) - (w+1)\sigma_\lambda\nabla_{z_\lambda}\log{p_{\phi}}(c|z_\lambda)$  
$\simeq -\sigma_\lambda \nabla_{z_\lambda}[\log{p(z_\lambda)} + (w + 1)\log{p_\theta (c|z_\lambda)}]$  
$= -\sigma_\lambda \nabla_{z_\lambda}[\log{p(z_\lambda | c)} + w \log{p_\theta (c|z_\lambda)}]$

와 같이 최종적으로 정리할 수 있다.

<img src="https://drive.google.com/uc?id=1n-wKgMDpK_VhxknTxUPVgZHAMpSAzSFW" height=200>

귀찮았던 classifier의 gradient가 없어졌다..!!  
(또한, w=-1이면 ddpm, w=0이면 conditional ddpm)

__결론__  
+ training : condition줬을 때의 노이즈와 uncondition일 때의 노이즈를 모두 학습시킨다.   
</br>
+ sampling : w 가중치를 통해 interpolation한 값을 사용한다.

__Q.그냥 condition 주고 u-net에 학습하면 안되나요? 뭐하러 두가지 나누고 interpolation하죠__

__A.기존에 그렇게 conditional ddpm을 해봤는데 학습이 잘 안됐다. 그래서 앞에서처럼 불편하게 classifier를 추가해준 것이다.__

최종적인 학습 형태이다.  
<img src="https://drive.google.com/uc?id=1YJ_vcQDRE4xue1VN6b1WJe_IxwspGzhI" height=230>

p라는 확률 값을 통해, condition을 랜덤하게 주거나 안주면서 학습시키고, (사실상 원래의 conditional ddpm 코드에 $p$값 추가해주는 한줄 추가해주면 된다.)

샘플링 알고리즘이다.  
<img src="https://drive.google.com/uc?id=13W6-zosvA-uomQa0XvHHHvcy1ynAayBI" height=280>

<img src="https://drive.google.com/uc?id=1lKoQ-Aevs2EWox29nqOzggNCPel0b8Sz" width=700>

즉, conditional 모델에 unconditional모델을 이용해 $w$만큼 guidance를 준다고 볼 수 있다.

<img src="https://drive.google.com/uc?id=1uptoXpcfqdk5vEMvgO4-qNiqjFVlO1sf" width=600>  
w 값을 올릴수록, FID 증가, IS 증가하는 모습을 볼 수 있다.(Trade off)

<img src="https://drive.google.com/uc?id=1ETzX7C5vm8Naa31pIHTgB-OC5mbIfVl_" height=300>
