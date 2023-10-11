# cechallenge
https://cechallenge.github.io/   

사용한 환경이 권한상 docker를 사용할 수 없는 환경이라 conda를 사용함   
conda 환경은 yaml파일로 업드로 해두었음 [링크](https://github.com/abzb1/cechallenge/blob/main/cechallenge.yaml)   

기존 Llama v1 30B의 4-way TP 가중치를 조립 후 해체하여 4-way PP로 구성
HellaSwag dataset Validation set에 대해 1.44x 가속(5914s -> 4111s)   

### 코드 사용법   
conda 환경에서 torchrun.sh 파일을 이용하여 멀티프로세스 스폰   
각각 inference_base의 torchrun.sh과 inference_pipelined의 torchrun.sh로 구성   
실행인자에 data-path, tokenizer-path, ckpt-path를 명확히 지정하면 됨 

### PP 가중치
알아서 만들 것
