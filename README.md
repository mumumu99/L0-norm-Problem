# L0-norm-Problem
Solve L0-norm problem

주어진 data의 각 sample은 1 또는 -1을 원소로 갖는 n-dim binary vector(np.array)와 one-hot vector로 표현된 label로 이루어져 있습니다. 
(n = 12)

XOR 연산을 일반화할 수 있는 여러 문제들 중 하나는 홀/짝 문제입니다.  
**$n$개의 원소를 갖는 1차원 bool 벡터의 $L_0-norm$의 값이 홀/짝인지 판단하는 문제**에서, $n=2$일 때, XOR 연산과 동일합니다.  
(여기서 $L_0-norm$은 벡터 내에 0이 아닌 원소의 갯수를 의미합니다.)  

**PyTorch를 이용하여 주어진 data에 대해 위 문제의 답을 출력하는 머신을 지도학습 기반으로 학습시켜주세요.**
  
$2^n$개의 samples를 갖고있는 dataset $U$를 가정합니다. i.e. $|U|$ = #($U$) = $2^n$  
such that #(train) + #(test) + #(validation) = $2^n$ and train $\cup$ test $\cup$ validation = $U$
