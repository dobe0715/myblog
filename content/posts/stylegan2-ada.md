+++
title = "Training Generative Adversarial Networks with Limited Data (stylegan2-ada) review"
+++

# Review : Training Generative Adversarial Networks with Limited Data (stylegan2-ada)
+ ADA(Adaptive Discriminator Augmentation)
+ 필요한 데이터 수십, 수백만장 -> 수 천장

## GAN 데이터 많이 필요한 이유
+ discriminator가 적은 양의 데이터 셋에 과적합 될 수 있다.

<img src="https://drive.google.com/uc?id=1sKMO7OlEjd3-Z9BYhnjo-8cDO53uZJ6a" height=250>  

+ (a)를 보면, 14만장이 모여야 과적합 발생하지 않음을 볼 수 있다.
+ (b), (c)를 보면, fid score로부터 계속 멀어짐을 알 수 있고, 검증데이터와 생성데이터는 가깝게 판단하는 반면에, 훈련데이터는 반대로, 완전히 과적합 상태에 가까워진다는 것을 알 수 있다. 무려 2만, 5만인데도..
+ __만약 검증이랑 실제데이터는 증가하고 생성이 내려갔으면 generator가 약하구나.. 라고 생각하겠지만 그렇지 않고 확실하게 과적합 상황.__


## Stochastic Discriminator Augmentation

<img src="https://drive.google.com/uc?id=15XlfwqqSgINcEHz9Au_Sor0gKNrytY8N
" height=270>  
(회색부분은 gradient descent안할 때임. 즉, 학습하지 않는다.)
+ CR : input data에 randomize하게 augmentation을 진행시켰을 때, aug시킨거랑 안시킨거 두개의 결괏값에 생기는 차이 최소화 시키는 기법.

+ 기존방법(bCR) : balanced Consistancy Regulerization
    + real이미지, generated이미지 둘다 consistancy regulerization을 진행
        
+ 기존의 bCR방법의 augmentation은 generator가 penalty없이 증강데이터를 만들어버려 문제가 생길 수 있다. 즉, 생성이미지로 증강데이터가 누출되어버린다.

## Adaptive Discriminator Augmentation
+ 기존의 방법은 augmentation을 정해진 최적화된 수식에 따라 수행했다.
+ 본 논문에서는 discriminator가 과적합되는 정도에 따라 동적으로 augmentation을 수행한다. (p)
+ 또한, 생성자 학습에도 동일하게 적용한다.(어차피 누출될거라면 동적으로 제어해버리겠다)

<img src="https://drive.google.com/uc?id=1AYRv1-_kfhUvArYq8t4oBinejynSmayv" height=400><img src="https://drive.google.com/uc?id=1b9D_EKQUEAvdbU7V0Y3NXUZ1_fDOhxP8" height=80>

$r_v$가 0에서 1에 가까워질수록 overfitting이 일어난다고 볼 수 있다.  
$r_t$는 real데이터에대해 discriminator가 양의 값인지, 음의값인지의 비율을 의미한다.

+ $E(D_{train})$를 바로 계산하는것보다 더 효율적이었다고함.

+ 이제, p값을 0~1사이로 준다.(각 epoch마다 0으로 초기화) 
    + 배치마다 학습을 하면서, 
    + 과적합이 많이된다 -> p 증가
    + 과적합이 덜된다 -> p 감소



#### p값을 augmentation에 어떻게 사용하는가

<img src="https://drive.google.com/uc?id=1cg3Q5SeeB5h4CNYEERNL2M5mBD1aPcmB" height=300>  
+ 각 augmentation마다 leaking되는 p값이 다르다... -> random하게 augmentation방법을 선택해서, 각각의 방법에 따라 p값이 다 보존되어있고, 사용할때마다 갱신하는방식

<img src="https://drive.google.com/uc?id=1E8svY9VD0zHnIVvkfdyaBMipWNaED4oa" height=250>

+ 기존에 비해서...
    + FID 점수 잘나옴
    + 오버래핑 잘됨
    + 기울기 갱신이 잘 됨
