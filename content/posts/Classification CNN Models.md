+++
title = 'CNN 모델 정리'
+++

# Inception Net(GoogLeNet)  
+ https://arxiv.org/pdf/1409.4842.pdf
+ 참고 블로그  
    + https://phil-baek.tistory.com/entry/3-GoogLeNet-Going-deeper-with-convolutions-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0

## 모델 나오게 된 배경

__DNN의 성능을 향상시키는 방법 가장 직접적인 방법은 size를 늘리는 것에서 시작__

구체적으로, depth/width를 증가시키면 되는데 이 때 두가지 문제점이 있다.


1. 파라미터 수의 증가  
파라미터가 늘어날 경우, 이에 비하여 학습데이터가 부족하면 오버피팅에 일어나기 쉽다.  


2. 컴퓨팅 자원 사용량의 증가  
만약, 두개의 conv layer가 연속으로 있는 상황에서, 거의 대부분의 가중치가 0이면, 계산에 있어서 버려지는 자원이 많아진다.  
__but, 컴퓨팅자원은 한정적이므로 효율적으로 분배하는 것이 중요__

__해결방법으론, dense한 Fully Coneected에서, Sparsely Connected(연관있는 애들 끼리) 구조로 바꾸는 것.__   
(대부분의 경량화는 결국 이것이 목표인 듯 하다.)  
<img src="https://drive.google.com/uc?id=1gu66JQDTTMD2_5AdAnWiqToDfhYPV3Zl" height=200>  

왼쪽이 sparsely, 오른쪽이 fully

__but, sparsely한 구조를 사용하려면 더 섬세한 작업이 필요하고, 이러한 것은 컴퓨팅의 병렬 작업에 그리 적합하지 않다.__

그래서, Sparse한 효과를 내는 비슷한 구조를 만들어내는 것이 목표이다

## 모델 구조

<img src="https://drive.google.com/uc?id=1woV11XSA4YXh-JZMlfroWXhSFd_JJn0T" height=250>

<img src="https://drive.google.com/uc?id=1vwaJCTZlR1YsZ_DUIN0w9rpn2x8H9S9S" height=250>

filter의 size에 따라 잡아낼 수 있는 local한 의미가 다르기 때문에, 각각이 반응하도록 하고, 이를 통해 sparse한 효과를 낼 수 있다.

그리고, 병목현상 및 연산량을 줄이기 위해서 각각의 3x3, 5x5 전에 1x1 layer를 추가해주었다.

max pooling 이후에 1x1한 것은 previous layer의 특징을 보존시키기 위해서라고 생각이 든다.

어쨋든, GoogLeNet은 이러한 inception module을 여러개 쌓아서 만든 구조이다. 이후의 detail한 부분은 읽어보면 될정도

# ResNet
+ 논문
    + resnet : https://arxiv.org/pdf/1512.03385.pdf
    + resnet ensemble : https://arxiv.org/pdf/1605.06431.pdf
+ 참고 블로그
    + 투빅스 : https://tobigs.gitbook.io/tobigs/deep-learning/computer-vision/resnet-deep-residual-learning-for-image-recognition
    + https://blog.kubwa.co.kr/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-deep-residual-learning-for-image-recognition-2015-%EA%B0%84%EB%8B%A8%ED%95%9C-%EC%8B%A4%EC%8A%B5%EC%BD%94%EB%93%9C-a55cb68981b1

+ 참고 유튜브
    + 혁펜하임 : https://www.youtube.com/watch?v=Fypk0ec32BU&list=PL_iJu012NOxd_lWvBM8RfXeB7nPYDwafn&index=14&t=0s

## 모델 나오게 된 배경
__depth가 단순히 깊어지기만 해서 모델이 좋아지지는 않는다.(순전히 오버피팅이 아닌게, 훈련 셋의 정확도 마저도 낮다.)__  
<img src="https://drive.google.com/uc?id=1Msrl-fYQ2cdfjbDfTCuDAubKH93WRw5w" height=150>

## 모델 구조

아래의 block을 마구마구 쌓은 구조를 갖는다.  
<img src="https://drive.google.com/uc?id=1Axk9uD7YpO-Vo_qshUNxdbC2rxoyedIH" width=400>

논문에서 말하고자하는 바는 다음과 같다.  
우리가 얻길 원하는 이상적인 함수 $H(x)$가 있다고하면, $F(x)$를 다음과 같이 정의한다.

$F(x) := H(x) - x$

이 때, 이상적인 함수가 $H(x) \simeq x$를 만족하면 좋겠다고 이야기함.

__why? 아마도, layer가 깊어지면 깊어질 수록 값들 변화하는 횟수가 많아지는데, 그 변동 폭이 너무 크다면 오버피팅의 위험도 있고, 학습하는 것에 어려움이 있을 것이다.__

즉, identity 함수와 비슷해지도록 학습하는 것이 좋다..!!  
<img src="https://drive.google.com/uc?id=1pt10hrb5jnuM-Q45ySCpcghwebsuJsOf" width=400>

가정 : non linear layer들이 점진적으로 복잡한 함수로 접근해나갈 것이다.($H(x) \simeq x$)

그렇다면, 이러한 상황에서 residual connection을 하면 왜 학습이 잘되는가?

__non linear layer를 포함한 함수 $F(x)$를 가지고, identity mapping을 하는 것보다, zero mapping을 하도록 optimizing하는 것이 더 쉽기 때문이다.__

선형 레이어들의 경우에는 초깃값을 단위행렬 이런거로 맞추면 가능하겠지만, 비선형 레이어의 경우에는 초깃값을 어떻게 설정할지도 의문이고, 그럴 바에야 어차피 초깃값 0 근처로 잡아버리고 그 근처로 optiminze하는 것이 훨씬 쉽다.


__Residual Learning!__

<img src="https://drive.google.com/uc?id=15Hj6l_VNuoL28PeI_HoTVitNH9c5jUkn" height=300>

기존의 plain network의 경우에는 층이 깊어졌을 때, error가 증가하는 모습을 볼 수 있다.   
하지만, resnet의 경우에는 층이 깊어지면 error가 내려간다.

## ResNet의 다른 해석(앙상블)
+ (2016)Residual Networks Behave Like Ensembles of
Relatively Shallow Networks

<img src="https://drive.google.com/uc?id=1V-gPfItp_Cb0HM6GJPYQzR6YxYKxAxnF" height=250>

<img src="https://drive.google.com/uc?id=1vPqH0U2gdOEA4Ds47EhHMDyZbi_RL9r9" height=280>

이처럼 앙상블로 볼 수 있다는 이야기와, layer들을 몇개 제거해봤을 때,   
resnet은 성능에 큰 차이가 없으나 vggnet은 거의 제대로된 기능을 하지 못한다.

# Mobile Net
+ 논문
    + https://arxiv.org/pdf/1704.04861.pdf
+ 참고 블로그
    + https://deep-learning-study.tistory.com/532
    + https://velog.io/@twinjuy/MobileNet-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0

## 모델 나오게 된 배경
__mobile, embedded vision network에 적용하기 위해 모델을 경량화 시켜야 한다.__  
즉, 저용량 메모리 환경에서 accuracy와 latency(깊이)의 균형을 잘 찾기 위함

## 모델 구조


__Depthwise Separable Convolution__

<img src="https://drive.google.com/uc?id=1_rbOTUyYzxpKNUu94qXjmMrf35Ww4HvL" height=400>

해당 모델은 Depthwise Convolution을 거치고, Pointwise Convolution을 결합한 것이다.

<img src="https://drive.google.com/uc?id=1W52fKDJY4urIA93aRedelcM1zBNJTUAr" width=300>

(a)를 보면, 일반적인 conv filter이다.   
kernel size를 $D_K \cdot D_K$이라하고, input channel을 M, output channel을 N이라하고 결과 feature map size를 $D_F \cdot D_F$라 하면 다음의 컴퓨팅 cost가 나온다.  
$$D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$$

(b)는 depthwise conv filter이다.  
$D_K \cdot D_K \cdot 1$size의 filter가 이전 layer의 각각의 channel에 conv 연산을 수행해준다. 따라서 컴퓨팅 cost는 다음과 같다.
$$D_K \cdot D_K \cdot M \cdot D_F \cdot D_F$$

(c)는 pointwise conv filter이다.  
이전 layer에 대해서 M의 input channel, N의 output channel을 얻도록 해주려 한다. 또한 feature map의 크기가 $D_F \cdot D_F$로 나오고 1x1으로 곱해주므로 다음의 컴퓨팅 cost를 갖는다.
$$M \cdot N \cdot D_F \cdot D_F$$

최종적으로 classic한 conv layer와의 computation cost를 비교해보면 다음과 같다.  
$$\cfrac{D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F}{D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F} = \cfrac{1}{N} + \cfrac{1}{D_K^2}$$


3x3 kernel을 쓰게되면, 기존에 비해 8~9배정도 computation을 덜 하게 된다고 한다.

여기에 추가적으로 width/resolution Multiplier를 적용해 더욱 경량화를 시킬 수 있었다. 최종적인 식은 다음과 같다.
$$D_K \cdot D_K \cdot \alpha M \cdot \rho D_F \cdot \rho D_F + \alpha M \cdot \alpha N \cdot \rho D_F \cdot \rho D_F$$

$\alpha$는 0.25, 0.5, 0.75, 1 으로 width multiplier   
$\rho$는 resolution multiplier   
각각 $\alpha^2, \rho^2$만큼의 computational cost를 절약시켜줬다.

<img src="https://drive.google.com/uc?id=11U3QNm_2Z2E6-JH_gdkH4jHvJtCIPKFh" height=150>

정확도는 기존의 GoogLeNet과 VGG16과 비교해 크게 차이 없었고, 연산량은 확연히 줄어들었음을 알 수 있다.

# SeNet
+ 논문
    + https://arxiv.org/pdf/1709.01507.pdf
+ 참고 블로그
    + https://inhovation97.tistory.com/48
    + https://deep-learning-study.tistory.com/539

## 모델 나오게 된 배경

__feature map의 채널마다의 relationship을 이용하면 성능을 끌어올릴 수 있지 않을까?__

## 모델 구조

__Squeeze & Excitation__

<img src="https://drive.google.com/uc?id=1qGzRfGuVkX2GGEfWXHh6rFB0T6POfuyq" height=250>

__1. Squeeze(Global Information Embedding__)  
기존의 input $X$에 $F_{tr}$라는 연산을 거쳐 $H \cdot W \cdot C$의 형태를 가진 feature map($U$)에 대해서 각각의 channel에 대해 squeeze(__global average pooling__) 작업을 해준다.



수식으로 표현하면 다음과 같다.
$$z_c = F_{sq}(u_c) = \cfrac{1}{H * W}\sum_{i=1}^H \sum_{j=1}^W{u_c(i, j)}$$

__2. Excitation(Adaptive Recalibration)__  
앞의 squeeze에서 모아놓은 정보에다가 채널마다의 정보 의존성을 넣는다. 두가지 조건을 만족해야 하는데, flexible해야하고 non mutually exclusive해야 한다.(유연 + 상호의존적) 그러기 위해, fc-relu-fc-sigmoid 순으로 layer를 거쳐준다.  
이 때, reduction ratio(r)만큼으로 bottle neck크기를 설정한다.(16으로 제안)

수식으로 표현하면 다음과 같다.  
$$s = F_{ex}(z, W) = \sigma(g(z, W)) = \sigma(W_2\delta(W_1z))$$

마지막으로 원래의 feature map $u_c$에 곱해주어 결론적으로 각각의 feature마다의 중요도? 만큼씩 scaling된다고 볼 수 있다.


$$\tilde{x}_c = F_{scale}(u_c, s_c) = s_cu_c$$

즉, 짜내서 feature 정보를 embedding하고 반응성에 따라 재교정한다 => Squeeze & Excitation  

__think__  
어떻게 보면, feature map에 대해서 self-attention하는 것과 비슷한 효과를 내는 것같다.(channel끼리 내적하는 대신에 layer들을 통해 channel끼리의 관계를 파악한다)

<img src="https://drive.google.com/uc?id=1n0iHhdqQEd2AbSe9KQG8XkaXKIspAy7I" height=300>

<img src="https://drive.google.com/uc?id=1psg512DrCtimw3IQW-x_INVdlvwTW7bu" height=200>  
기존의 모델에 추가했을 때 성능이다. 전체적으로 모두 좋아짐을 알 수 있다.  

# Efficient Net
+ 논문
    + https://arxiv.org/pdf/1905.11946.pdf
+ 참고 블로그
    + https://blog.kubwa.co.kr/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-efficientnet-rethinking-model-scaling-for-convolutional-neural-networks-2019-%EA%B0%84%EB%8B%A8%ED%95%9C-%EC%8B%A4%EC%8A%B5%EC%BD%94%EB%93%9C-cbe0e9963ffc

## 모델 나오게 된 배경

__기존에 나온 모델의 size가 과연 무엇이 최적일까__

<img src="https://drive.google.com/uc?id=1GdKGokQtrrI6hZ2aUYNmoScOMW082-B-" height=400>

해당 논문에서 크게 3가지 scaling에 초점을 맞춘다.  
<img src="https://drive.google.com/uc?id=1ASBCpZJTarB9cegre_Z6NckhyHjbNtOP" width=350>

<img src="https://drive.google.com/uc?id=1iKkBYof6LN6lk_l6esxoghTEiG1aCSQZ" height=300>

1. depth(모델 깊이)  
기존에 ResNet같은 경우 ResNet-18 ~ ResNet-1000 까지 많은 층을 쌓으며 모델의 성능을 끌어올렸음 을 알 수 있다. 그런데, 성능을 보았을 때, 101층짜리와 1000층짜리의 정확도에서 큰 차이가 없었다.

2. width(채널 수)  
주로 작은 모델에 대해 사용하는 scaling인데, 마찬가지로 3.8과 5.0을 비교해보면 큰 차이가 없다

3. resolution(이미지해상도)  
마찬가지로 해상도가 높아질수록 성능은 올라가지만 그 정도가 점점 감소함을 알 수 있다.

## 모델 구조

<img src="https://drive.google.com/uc?id=1_2U8fgiIEy5G8iLHbdKxoJT1oTSn9DCq" height=280>
<img src="https://drive.google.com/uc?id=1l87bwW08gM3ITZIGrfpDLPvjNp4KIx7p" height=150>

Basline network이 되는 EfficientNet-B0부터 시작하여 오른쪽의 수식에 따라 d, w, r을 바꿔가며 모델을 실험해나간다.

이 때, MBConv는 Mobile block을 의미하고  
각각의 layer마다 SE 작업을 해주었다.

<img src="https://drive.google.com/uc?id=1HSVpyQzs2jpLrShYm_EgryhdSLfksdUP" height=200>

1. 3가지 하이퍼파라미터에 대해 기준을 두고 grid search를 한다.
2. 그 값을 기준으로 모델의 전체적인 크기를 늘린다.

즉, 최적의 모양을 먼저잡고 그것을 동일한 비율로 잡아 늘렸다고 보면된다.  
(물론, 각각의 3가지 scale이 독립적이지 않아서 이렇게 늘리는 것이 최선은 아니겠지만, test하는 것에 시간이 너무 오래 소요되어서 차선으로 이 방법을 택한 듯 하다.)

<img src="https://drive.google.com/uc?id=1PPKyxTg_bS4RtikbaZkvKdrkxUH-33y0" height=500>

기존의 비슷한 성능을 내는 network들과 비교했을 때 parameter수와 FLOPs(계산량)수가 현저히 낮고, B7 모델은 sota를 달성했다.

<img src="https://drive.google.com/uc?id=1WmxjtKAzBmoK-TYtkYgTy2teELoIrN9d" height=450>
