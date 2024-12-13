# IITP Report

한미공동연구(IITP) 성능 보고용 레포지토리

## Quickstart

### Step 1. Setup
```bash
git clone https://github.com/seoultech-HAILAB/IITP-report.git

cd IITP-report
conda activate iitp-report      # 114.71.19.168에 이미 구축되어 있는 가상환경

# (Optional) OpenAI API 사용할 경우
cp .env.example .env            
```

### Step 2. Prepare dataset
- From [Huggingface](https://huggingface.co/datasets?sort=trending)
- From ...

> ※ 다른 소스로부터 데이터셋 가져와야 할 경우, 별도의 DataLoader 구현 필요

### Step 3. Run evaluation
```bash
# OpenAI API
python run_evaluation.py --model gpt-4o-mini --api_type openai

# Ollama (local LLM)
python run_evaluation.py --model llama3.1:70b --api_type ollama
```

## Usage Example
```bash
# 로컬 LLM의 맞춤형 콘텐츠(퀴즈) 생성 성능 평가
python run_evaluation.py \
--domain_type DT \           # DT: 디지털교과서, DP: 토론프로젝트
--task_type QG \             # QG: 퀴즈 생성, ...
--datasource huggingface \
--dataset_name lmqg/qg_squad \
--temperature 0.1 \
--model llama3.1:70b \
--api_type ollama \
--api_key {YOUR_API_KEY} \
--eval_type reference_based \
--reset False
```

## Evaluation Results
- [BLEU](https://huggingface.co/spaces/evaluate-metric/bleu/blob/main/README.md) (higher score is better)
- [CER](https://huggingface.co/spaces/evaluate-metric/cer) (lower score is better)
- [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge/blob/main/README.md) (higher score is better)

> ※ 참고
> - BLEU의 n값은 1로 설정
> - CER은 SER의 대체 지표
> - ROUGE 스코어는 ROUGE-L 기준

| 구분 | 평가항목 | 성능지표 | 목표치 | 결과 |
| --- | --- | --- | --- | --- |
| 1차년도 | 디지털교과서 AI 역량 분석 오차 | MAE | $\leq1$ |  |
|  | 토론프로젝트 AI 역량 분석 오차 | MAE | $\leq1$ |  |
|  | AI 역량 분석 코파일럿 작업부하 | NASA-TLX | $\leq50$ |  |
| 2차년도 | 디지털교과서 AI 맞춤형 콘텐츠 생성 적합성 | BLEU | $\geq0.5$ | **0.3039** |
|  | 디지털교과서 AI 맞춤형 콘텐츠 생성 오류율 | SER | $\leq0.01$ |  |
|  | 디지털교과서 AI 맞춤형 콘텐츠 생성 재현성 | ROUGE | $\geq0.5$ | **0.3414** |
|  | 토론프로젝트 AI 맞춤형 콘텐츠 생성 적합성 | BLEU | $\geq0.6$ |  |
|  | 토론프로젝트 AI 맞춤형 콘텐츠 생성 오류율 | SER | $\leq0.01$ |  |
|  | 토론프로젝트 AI 맞춤형 콘텐츠 생성 재현성 | ROUGE | $\geq0.5$ |  |
|  | AI 맞춤형 콘텐츠 생성 코파일럿 작업부하 | NASA-TLX | $\leq50$ |  |

## Code Structure
```bash
IITP-report/
├── data/                                               
│   ├── raw/                  # 다운받은 데이터셋 (DataLoader의 출력, PayloadCreator의 입력)
│   └── preprocessed/         # API 요청 리스트 (PayloadCreator의 출력, APIExecutor의 입력)
├── lib/
│   ├── data_loader.py          
│   ├── payload_creator.py      
│   ├── api_executor.py         
│   ├── response_evaluator.py            
│   └── utils.py
├── prompts/                  # 시스템 프롬프트
├── results/            
│   ├── output.jsonl          # LLM 응답 리스트 (APIExecutor의 출력, ResponseEvaluator의 입력)
│   ├── .eval_results.jsonl   # 응답 평가 결과 (ResponseEvaluator의 출력)
│   └── .eval_summary.json    # 응답 평가 결과 요약 (평균 점수)
└── run_evaluation.py
```