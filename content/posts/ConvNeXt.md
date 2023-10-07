# A ConvNet for the 2020s

참고링크
+ https://arxiv.org/abs/2201.03545
+ https://americanoisice.tistory.com/121

## 논문의 아이디어
기존에 computer vision에서는 "sliding window"기법이 주된방법이었다.
반면에 NLP분야에서는 transformer라는 엄청난 모델이 나왔고, 이를 CV와 결합하여 ViT라는 모델이 나오면서, 이 역시 SOTA를 밥먹듯이 하고있다..  
__Transformer에서 사용한 기술들을 ConvNet에 적용한다면 어떻게 될까?__

이 때, ResNet만 가지고 발전시킨다.

## ConvNet의 Modernizing

ResNet-50/200 vs Swin-T/B으로 모델을 대응시켜 각각 FLOPs를 맞춰서 비교한다.

<img src="https://drive.google.com/uc?id=1hZri47IZ7URnizOf2gHqLosAuPpMk1aK" width=400>

1) macro design 2) ResNeXt 3) inverted bottleneck 4) large kernel size 5) various layer-wise micro design

### 1. Training techniques

DeiTm, Swin Transformer에서 사용한 훈련 법을 가져왔다.
+ epochs 90 -> 300
+ AdamW optimizer
+ data augmentation(Mixup, Cutmix, RandAugment, RandomDrasing)
+ Stochastic Depth
    + layer를 무작위로 drop해준다.(Drop out의 layer 버전이라고 볼 수 있음. test때에는 dropout과 마찬가지로 1-p곱해주는 형식)
+ Label Smoothing

얘네들 사용해서 76.1% -> 78.8%(+2.7%)의 상승효과 얻었다.

<img src="https://drive.google.com/uc?id=1Vd9lkFwKZV_sqZIOZn0duJXgpIwZyHc4" width=600>

### 2. Macro Design
Cell 단위의 조정이다.

+ __compute ratio 수정__  
기존의 ResNet은 각 stage마다 3:4:6:3번의 연산을 반복했다.   
but, swin transformer에서는 가장 작은건 1:1:3:1, 그 이후는 1:1:9:1 비율의 ratio로 연산을 반복하였다.  
그래서 마찬가지의 형태를 적용.
78.8% -> 79.4% 상승

+ __Stem cell을 Patchify하게 바꾸기__  
기존 ResNet은 7x7(2), max pool(2)를 통해 input image를 4배 downsampling하였다.  
이에 반해, ViT에서는 patch단위 14 혹은 16를 기준으로하였는데, 이는 non-overlapping conv를 사용했다고 볼 수 있다.
</br>  
따라서, 4x4(4)를 사용하였다.  
79.4% -> 79.5% 상승

### ResNeXt-ify
ResNext에서 사용한 grouped conv의 개념을 이용한다.

<img src="https://drive.google.com/uc?id=1SEBO0XcruizEQRoOHphE_0J3xATjIIsl" height=350>  
앞의 feature map으로부터 group 단위로 담당해서 conv연산을 수행한다!

실제로, ResNeXt에서는 conv block을 group conv로 바꾸는 대신, 그만큼 channel 수를 늘려서 bottle neck을 완화하고 같은 parameter수 대비 성능을 끌어올렸다.

이 때, 특별하게 group수와 앞의 feature map channel수가 동일하다면, 각 channel마다 담당하는 conv block이 생기게 되는데 이는 정확히 depthwise convolution과 동일하다.(Mobile Net)  
<img src="https://drive.google.com/uc?id=1_rbOTUyYzxpKNUu94qXjmMrf35Ww4HvL" height=400>

해당 논문에서는 이러한 depthwise convolution이 self-attention에서의 weighted sum operation과 유사하다고 이야기한다.

<img src="https://drive.google.com/uc?id=1x1h2kDRjcfaTHuSwQ9-GT1yOu0zC0NGx" height=300>  
의미를 해석해봤을 때, 각 feture map의 channel마다의 pixel을 기준으로, 주변정보와 self-attention했다고 생각할 수 있어보인다.

어쨌든, 이렇게 depthwise conv이후에 1x1 conv를 통해 channel mixing까지 적용한 모델을 사용하였다. 그리고, Swin-T에서의 channel수와 동일하게 96을 적용하였으며   
그 결과 80.5%의 퍼포먼스를 보였다.

### 4. Inverted Bottleneck

<img src="https://drive.google.com/uc?id=1_BNXlSrNFCX012mBwIBp4rsvWfhRzHW7" height=200>

(a)가 기존의 ResNeXt에서 사용한 구조이다. 우리는 depthwise seperable block을 사용하기 때문에,  
 중간의 bottle neck(3x3 conv연산 하는 구간)을 오히려 크게 키워도 연산량을 커버할 수 있게 된다.  
따라서, (b)가 바꾼 형태이고, (c)와 같이 앞으로 옮겨서도 실험해봤는데(Transformer의 MSA block모양을 따라하기 위해) 성능이 하락하는 모습을 보였다..  
(b)를 적용했을 때,  
80.5% -> 80.6%(Res-200에 적용하면 81.9% -> 82.6%)

### 5. Large Kernel Sizes


+ depthwise conv layer 이동
ViT 모델의 MSA block과 형태를 동일하게 하기 위해서 conv 블럭을 앞으로 뺐다.(MSA는 self-attention 이후 mlp)
즉, (c)와 같은 형태처럼 conv 이후 1x1연산 함으로써 mlp 보낸 효과.  
but, 성능은 79.9%으로 하락하였다.

+ kernel size 증가  
ViT모델의 경우, non-local self-attention을 적용하기 때문에,(patch단위로 전체 attention) global한 receptive field를 가진다. 이것이 성능에 중요한 영향이라고 생각하여, conv size늘려보겠다.  
이 대, 3, 5, 7, 9, 11에 대해 FLOPs를 고정시키고 바꾸어 봤는데, 7일 때 가장 좋았다.  
성능은 79.9% -> 80.6%으로 향상되었다.

__Think__  
그러면 depthwise 이동안시키고 kernel size증가는 안시켜봤나??

### 6. Micro Design
activation, normalization을 어떤걸 사용할지 선택

<img src="https://drive.google.com/uc?id=1yaTHHYwIxfJlj8LFwbHMVeDqr2GmFVoF" width=500>  
최종적인 ConvNeXt block의 형태이다.

+ ReLU -> GELU  
최근의 많은 연구에서 ReLU대신에 GELU를 사용했을 때, 좋은 효과를 봤다. 그래서 마찬가지로 GELU로 바꿔봤다.  
80.6% 동일.

+ activation 줄이기  
Transformer의 경우, MLP사이에  GELU 하나 뿐이다.  
마찬가지로, 1x1 블럭 사이에 GELU만 추가한다.(이것 때문에 위에서 depthwise 이동시킨 것 같다.)

+ BN -> LN  
Batch normalization은 모델에 있어서 꽤나 복잡한 악영향을 끼치기도 한다.(주로 데이터 상의 문제)  
따라서, transformer의 경우 LN을 사용해왔는데, 여기에서도 BN을 LN으로 바꿔보았다. 7x7 conv 직후에 넣었다.  

accuracy 81.5%으로 사응하였다.

+ downsampling layer를 분리해보기(block으로부터)  
Resnet의 경우, 3x3(2)를 적용한 후 shortcut connection을 진행할 때, 1x1(2) 적용해놓고 합쳤다.(정보의 손실발생..)  
하지만, Swin-T의 경우, 각 stage사이에 downsampling을 적용하였다.  
마찬가지로, 우리의 모델은 2x2(2) 를 각 block사이에 넣어준다.  
이를 마지막으로 모델이 확정되어 LN이 들어가는 부분이 정해진다.
각 downsampling layer 이전 / stem 이후 / GAP이후  

accuracy 82.0%를 달성하였다.(Swin-T : 81.3%)


<img src="https://drive.google.com/uc?id=1pVdvy_OlrQp3ZoeanwBq_usrlaXK6a64" width=450>

기존 ViT모델 에 대해 비슷한 param, 연산량 대비 정확도가 더 높고 처리속도가 빠르다.
