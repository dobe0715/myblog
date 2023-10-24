+++
title = "VAE 정리본"
+++

# VAE
(송경우교수님 딥러닝강의 2022) https://www.youtube.com/watch?v=V-lWbJtNzTc&list=PLeiav_J6JcY8iFItzNZ_6PMlz9W4_jz5J&index=58

<img src="https://drive.google.com/uc?id=1d4eR5ta0-gOjods4S60vOyR_IUb-oLuD" height="300">

## Generative model

+ MLE의 관점에서 보았을 때, $p_\theta(x)$를 최대화 하는 파라미터를 찾는 것이다!

결국 목적함수는,   
$$\theta^* = \arg\max\limits_{\theta}\cfrac{1}{N}\sum_{i=1}^N\log{p_\theta(x_i)}$$  

이다.

#### Variational Inference
+ 어떤 조건이 주어졌을 때의 확률($p(z|x)$)을 다루기 쉬운 확률분포($q(z)$)로 근사하는 것.

<img src="https://drive.google.com/uc?id=1ryponrQU_kCjVlBcexogjeET47CrquS9" width=600>

<img src="https://drive.google.com/uc?id=1moaCYsW9Tyy0SDBFMTNZtXXTfQ3wjaZ_" width=800>

즉, log-likelihood($\log{p(x_i)}$)의 lower bound인 $L$를 maximize하면 결국에 $q_i(z)$가 $p(z|x_i)$와 가까워져, 실제 샘플에 대응하는 latent를 더 잘 뽑아줄 수 있게 된다.

이를 학습하기 위해 elbo를 maximize 하는 과정을 살펴보면,  
i번째 샘플 데이터학습할 때 마다, $q_i$로부터 $\mu_i, \Sigma_i$를 얻어내고, 여기로부터의 분포에서 다시 $\hat{x}$를 뽑아내서($\theta$), $x_i$와 닮도록 학습한다.  

이 때, 각 데이터 샘플마다 뮤와 시그마에 대응시키는 파라미터가 필요하다.  
-> 너무 많다... 새로운 데이터 들어올때마다 또 추가해야한다.

#### Amortized Variational Inference
+ $x_i$에 대응되는 $z_i$가 존재하는 상황
    + N개의 데이터가 있다면,,, 원래는 N개의 파라미터를 대응시켜서 학습시켰다..(n개의 튜플만들어서 각 튜플마다 파라미터로서 바꾸는느낌)
    + 이걸 차라리 하나의 네트워크를 통해 mapping을 시켜주겠다!!(보통 DNN에서 입력, 출력 원하는 방향으로 하듯이)

수식으로 보면, $q_i(z) \approx q(z|x_i)$ 왼쪽거 대신 오른쪽거로 VI를 하겠다는 뜻
데이터 하나마다 학습되는 과정을 수식으로 바라보면,  
$x_i$-> NN($q_\phi(z|x)$) -> $\mu(x_i), \sigma(x_i)$ -> $z=\mu(x_i) + \epsilon\sigma(x_i)$ -> NN($p_\theta(x|z)$) -> $\hat{x} \approx x$  
<img src="https://drive.google.com/uc?id=1d4eR5ta0-gOjods4S60vOyR_IUb-oLuD" height="300">

<img src="https://drive.google.com/uc?id=1y2FU4IzeWUANlmsDTDfc6rEsdfnaGJT6" height=400>

## 코드 실습


```python
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
```




    device(type='cuda')




```python
data_set = torchvision.datasets.MNIST('./data',
                                      train=True,
                                      transform=transforms.Compose([
                                          transforms.Resize((32, 32)),
                                          transforms.ToTensor()
                                          ]),
                                      download=True,
                                      )
```


```python
data_set
```




    Dataset MNIST
        Number of datapoints: 60000
        Root location: ./data
        Split: Train
        StandardTransform
    Transform: Compose(
                   Resize(size=(32, 32), interpolation=bilinear, max_size=None, antialias=warn)
                   ToTensor()
               )




```python
data_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=100)
```

### 모델구성
+ mnist data : 28x28
+ network :
channel : 4, 8, 16, 32 /
WxH : 28-14-7-4-2 -> fc1 4 -> 1(mu), fc2 4 -> 1(var)


```python
class VAE(nn.Module):

    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dims = None,
                 **kwargs):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i+1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1,
                      kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 128, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu

    def loss_function(self,
                      *args):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_loss

        return loss


    def sample(self,
               num_samples,
               current_device,
               **kwargs):

        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        return self.forward(x)[0]
```


```python
model = VAE(in_channels=1, latent_dim=200).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
```


```python
num = 0
for epoch in range(30):
    for i, (images, _) in enumerate(data_loader):
        # forward
        x = images.to(device)
        mu, log_var = model.encode(x)
        z = model.reparameterize(mu, log_var)

        x_rec = model.decode(z)

        # compute loss
        loss = model.loss_function(x_rec, x, mu, log_var)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"epoch : {epoch+1} | iter : {i} | loss : {loss:.4}")

        if i % 200 == 0:
            samples = model.sample(1, device)
            save_image(samples, f"./vae_samples/sample{num}.png")
        num += 1
```

    epoch : 1 | iter : 0 | loss : 25.12
    epoch : 1 | iter : 100 | loss : 0.2628
    epoch : 1 | iter : 200 | loss : 0.2504
    epoch : 1 | iter : 300 | loss : 0.1924
    epoch : 1 | iter : 400 | loss : 0.1042
    epoch : 1 | iter : 500 | loss : 0.2347
    epoch : 2 | iter : 0 | loss : 0.08283
    epoch : 2 | iter : 100 | loss : 0.06895
    epoch : 2 | iter : 200 | loss : 0.07953
    epoch : 2 | iter : 300 | loss : 0.07523
    epoch : 2 | iter : 400 | loss : 0.07775
    epoch : 2 | iter : 500 | loss : 0.08221
    epoch : 3 | iter : 0 | loss : 0.0736
    epoch : 3 | iter : 100 | loss : 0.06282
    epoch : 3 | iter : 200 | loss : 0.07073
    epoch : 3 | iter : 300 | loss : 0.05984
    epoch : 3 | iter : 400 | loss : 0.06127
    epoch : 3 | iter : 500 | loss : 0.06909
    epoch : 4 | iter : 0 | loss : 0.06461
    epoch : 4 | iter : 100 | loss : 0.06142
    epoch : 4 | iter : 200 | loss : 0.07161
    epoch : 4 | iter : 300 | loss : 0.05889
    epoch : 4 | iter : 400 | loss : 0.05847
    epoch : 4 | iter : 500 | loss : 0.06437
    epoch : 5 | iter : 0 | loss : 0.06036
    epoch : 5 | iter : 100 | loss : 0.06125
    epoch : 5 | iter : 200 | loss : 0.07911
    epoch : 5 | iter : 300 | loss : 0.06195
    epoch : 5 | iter : 400 | loss : 0.05938
    epoch : 5 | iter : 500 | loss : 0.06209
    epoch : 6 | iter : 0 | loss : 0.07897
    epoch : 6 | iter : 100 | loss : 0.06053
    epoch : 6 | iter : 200 | loss : 0.07334
    epoch : 6 | iter : 300 | loss : 0.07139
    epoch : 6 | iter : 400 | loss : 0.06082
    epoch : 6 | iter : 500 | loss : 0.05961
    epoch : 7 | iter : 0 | loss : 0.0998
    epoch : 7 | iter : 100 | loss : 0.05946
    epoch : 7 | iter : 200 | loss : 0.06843
    epoch : 7 | iter : 300 | loss : 0.06251
    epoch : 7 | iter : 400 | loss : 0.05751
    epoch : 7 | iter : 500 | loss : 0.05803
    epoch : 8 | iter : 0 | loss : 0.09001
    epoch : 8 | iter : 100 | loss : 0.05908
    epoch : 8 | iter : 200 | loss : 0.06798
    epoch : 8 | iter : 300 | loss : 0.05804
    epoch : 8 | iter : 400 | loss : 0.05706
    epoch : 8 | iter : 500 | loss : 0.05436
    epoch : 9 | iter : 0 | loss : 0.07401
    epoch : 9 | iter : 100 | loss : 0.0589
    epoch : 9 | iter : 200 | loss : 0.06516
    epoch : 9 | iter : 300 | loss : 0.05802
    epoch : 9 | iter : 400 | loss : 0.07783
    epoch : 9 | iter : 500 | loss : 0.05494
    epoch : 10 | iter : 0 | loss : 0.06096
    epoch : 10 | iter : 100 | loss : 0.05923
    epoch : 10 | iter : 200 | loss : 0.06546
    epoch : 10 | iter : 300 | loss : 0.05692
    epoch : 10 | iter : 400 | loss : 0.05633
    epoch : 10 | iter : 500 | loss : 0.05238
    epoch : 11 | iter : 0 | loss : 0.05586
    epoch : 11 | iter : 100 | loss : 0.05847
    epoch : 11 | iter : 200 | loss : 0.0642
    epoch : 11 | iter : 300 | loss : 0.05614
    epoch : 11 | iter : 400 | loss : 0.05614
    epoch : 11 | iter : 500 | loss : 0.05214
    epoch : 12 | iter : 0 | loss : 0.05507
    epoch : 12 | iter : 100 | loss : 0.05837
    epoch : 12 | iter : 200 | loss : 0.06383
    epoch : 12 | iter : 300 | loss : 0.05589
    epoch : 12 | iter : 400 | loss : 0.05605
    epoch : 12 | iter : 500 | loss : 0.05205
    epoch : 13 | iter : 0 | loss : 0.05482
    epoch : 13 | iter : 100 | loss : 0.05831
    epoch : 13 | iter : 200 | loss : 0.06379
    epoch : 13 | iter : 300 | loss : 0.05581
    epoch : 13 | iter : 400 | loss : 0.05601
    epoch : 13 | iter : 500 | loss : 0.05198
    epoch : 14 | iter : 0 | loss : 0.05462
    epoch : 14 | iter : 100 | loss : 0.05825
    epoch : 14 | iter : 200 | loss : 0.06375
    epoch : 14 | iter : 300 | loss : 0.05574
    epoch : 14 | iter : 400 | loss : 0.05597
    epoch : 14 | iter : 500 | loss : 0.052
    epoch : 15 | iter : 0 | loss : 0.05447
    epoch : 15 | iter : 100 | loss : 0.05825
    epoch : 15 | iter : 200 | loss : 0.06369
    epoch : 15 | iter : 300 | loss : 0.05568
    epoch : 15 | iter : 400 | loss : 0.05597
    epoch : 15 | iter : 500 | loss : 0.05191
    epoch : 16 | iter : 0 | loss : 0.0544
    epoch : 16 | iter : 100 | loss : 0.05823
    epoch : 16 | iter : 200 | loss : 0.06365
    epoch : 16 | iter : 300 | loss : 0.0556
    epoch : 16 | iter : 400 | loss : 0.05592
    epoch : 16 | iter : 500 | loss : 0.0519
    epoch : 17 | iter : 0 | loss : 0.0543
    epoch : 17 | iter : 100 | loss : 0.05822
    epoch : 17 | iter : 200 | loss : 0.06361
    epoch : 17 | iter : 300 | loss : 0.05559
    epoch : 17 | iter : 400 | loss : 0.0559
    epoch : 17 | iter : 500 | loss : 0.0519
    epoch : 18 | iter : 0 | loss : 0.05427
    epoch : 18 | iter : 100 | loss : 0.05824
    epoch : 18 | iter : 200 | loss : 0.06361
    epoch : 18 | iter : 300 | loss : 0.05553
    epoch : 18 | iter : 400 | loss : 0.05594
    epoch : 18 | iter : 500 | loss : 0.05185
    epoch : 19 | iter : 0 | loss : 0.05425
    epoch : 19 | iter : 100 | loss : 0.05821
    epoch : 19 | iter : 200 | loss : 0.06357
    epoch : 19 | iter : 300 | loss : 0.05552
    epoch : 19 | iter : 400 | loss : 0.05591
    epoch : 19 | iter : 500 | loss : 0.0519
    epoch : 20 | iter : 0 | loss : 0.05424
    epoch : 20 | iter : 100 | loss : 0.05821
    epoch : 20 | iter : 200 | loss : 0.06349
    epoch : 20 | iter : 300 | loss : 0.0555
    epoch : 20 | iter : 400 | loss : 0.0559
    epoch : 20 | iter : 500 | loss : 0.05188
    epoch : 21 | iter : 0 | loss : 0.05432
    epoch : 21 | iter : 100 | loss : 0.05819
    epoch : 21 | iter : 200 | loss : 0.06353
    epoch : 21 | iter : 300 | loss : 0.0555
    epoch : 21 | iter : 400 | loss : 0.05591
    epoch : 21 | iter : 500 | loss : 0.05189
    epoch : 22 | iter : 0 | loss : 0.05439
    epoch : 22 | iter : 100 | loss : 0.05821
    epoch : 22 | iter : 200 | loss : 0.06355
    epoch : 22 | iter : 300 | loss : 0.05548
    epoch : 22 | iter : 400 | loss : 0.0559
    epoch : 22 | iter : 500 | loss : 0.05187
    epoch : 23 | iter : 0 | loss : 0.05434
    epoch : 23 | iter : 100 | loss : 0.0582
    epoch : 23 | iter : 200 | loss : 0.06354
    epoch : 23 | iter : 300 | loss : 0.05552
    epoch : 23 | iter : 400 | loss : 0.05592
    epoch : 23 | iter : 500 | loss : 0.05189
    epoch : 24 | iter : 0 | loss : 0.0545
    epoch : 24 | iter : 100 | loss : 0.05819
    epoch : 24 | iter : 200 | loss : 0.06357
    epoch : 24 | iter : 300 | loss : 0.05547
    epoch : 24 | iter : 400 | loss : 0.05589
    epoch : 24 | iter : 500 | loss : 0.05189
    epoch : 25 | iter : 0 | loss : 0.05434
    epoch : 25 | iter : 100 | loss : 0.05818
    epoch : 25 | iter : 200 | loss : 0.06356
    epoch : 25 | iter : 300 | loss : 0.05543
    epoch : 25 | iter : 400 | loss : 0.0559
    epoch : 25 | iter : 500 | loss : 0.05189
    epoch : 26 | iter : 0 | loss : 0.0545
    epoch : 26 | iter : 100 | loss : 0.05818
    epoch : 26 | iter : 200 | loss : 0.06354
    epoch : 26 | iter : 300 | loss : 0.05548
    epoch : 26 | iter : 400 | loss : 0.05589
    epoch : 26 | iter : 500 | loss : 0.05191
    epoch : 27 | iter : 0 | loss : 0.05421
    epoch : 27 | iter : 100 | loss : 0.05817
    epoch : 27 | iter : 200 | loss : 0.06355
    epoch : 27 | iter : 300 | loss : 0.0555
    epoch : 27 | iter : 400 | loss : 0.05591
    epoch : 27 | iter : 500 | loss : 0.05192
    epoch : 28 | iter : 0 | loss : 0.05429
    epoch : 28 | iter : 100 | loss : 0.05816
    epoch : 28 | iter : 200 | loss : 0.06354
    epoch : 28 | iter : 300 | loss : 0.05548
    epoch : 28 | iter : 400 | loss : 0.0559
    epoch : 28 | iter : 500 | loss : 0.0519
    epoch : 29 | iter : 0 | loss : 0.05418
    epoch : 29 | iter : 100 | loss : 0.05819
    epoch : 29 | iter : 200 | loss : 0.06356
    epoch : 29 | iter : 300 | loss : 0.05547
    epoch : 29 | iter : 400 | loss : 0.05594
    epoch : 29 | iter : 500 | loss : 0.05189
    epoch : 30 | iter : 0 | loss : 0.05412
    epoch : 30 | iter : 100 | loss : 0.05817
    epoch : 30 | iter : 200 | loss : 0.06355
    epoch : 30 | iter : 300 | loss : 0.05546
    epoch : 30 | iter : 400 | loss : 0.05589
    epoch : 30 | iter : 500 | loss : 0.05189
    


```python

```


```python
# class VAE(nn.Module):

#     def __init__(self,
#                  input_feature,
#                  latent_dim,
#                  hidden_dim,
#                  **kwargs):
#         super(VAE, self).__init__()


#         # Build Encoder
#         self.e_fc1 = nn.Sequential(
#             nn.Linear(input_feature, hidden_dim),
#             nn.BatchNorm2d(hidden_dim),
#             nn.LeakyReLU(0.2)
#             )

#         self.e_fc2 = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm2d(hidden_dim),
#             nn.LeakyReLU(0.2)
#             )

#         # output layer
#         self.mu = nn.Linear(hidden_dim, latent_dim)
#         self.var = nn.Linear(hidden_dim, latent_dim)



#         # Build Decoder

#         self.d_fc1 = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim),
#             nn.BatchNorm2d(hidden_dim),
#             nn.LeakyReLU(0.2)
#             )

#         self.d_fc2 = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm2d(hidden_dim),
#             nn.LeakyReLU(0.2)
#             )

#         self.fianl_layer = nn.Sequential(
#             nn.Linear(hidden_dim, input_feature),
#             nn.Tanh()
#         )


#     def encode(self, input):
#         result = self.e_fc1(input)
#         result = self.e_fc2(result)

#         mu = self.mu(result)
#         log_var = self.var(result)

#         return [mu, log_var]

#     def decode(self, z):
#         result = self.d_fc1(z)
#         result = self.d_fc2(result)
#         result = self.final_layer(result)
#         return result

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return eps*std + mu

#     def loss_function(self,
#                       *args):
#         recons = args[0]
#         input = args[1]
#         mu = args[2]
#         log_var = args[3]

#         recons_loss = F.mse_loss(recons, input)
#         kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

#         loss = recons_loss + kld_loss

#         return loss


#     def sample(self,
#                num_samples,
#                current_device,
#                **kwargs):

#         z = torch.randn(num_samples,
#                         self.latent_dim)

#         z = z.to(current_device)

#         samples = self.decode(z)
#         samples = samples.view(-1, 28, 28)
#         return samples

#     def generate(self, x, **kwargs):
#         return self.forward(x)[0]
```
