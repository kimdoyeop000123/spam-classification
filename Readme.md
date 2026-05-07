# 📱 SMS 스팸 문자 분류 프로젝트

> TF-IDF 기반 전통 ML부터 BERT 파인튜닝까지 — 텍스트 분류 모델 비교 분석

---

## 📌 프로젝트 개요

SMS 문자 데이터를 활용해 스팸(spam)과 일반 문자(ham)를 분류하는 텍스트 분류 프로젝트입니다.  
TF-IDF 기반 전통 머신러닝 3개 모델과 BERT 파인튜닝을 비교하여 성능 향상을 검증했습니다.

---

## 🗂️ 데이터

- **출처**: UCI SMS Spam Collection Dataset
- **크기**: 5,572개 문자 메시지
- **구성**: ham(정상) 4,825개 (86.6%) / spam(스팸) 747개 (13.4%)

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| **언어** | Python |
| **전처리** | TF-IDF, Regex, BERT Tokenizer |
| **ML 모델** | Logistic Regression, Random Forest, XGBoost |
| **딥러닝** | BERT (bert-base-uncased), HuggingFace Transformers |
| **분석** | SHAP, Scikit-learn |
| **시각화** | Matplotlib, Seaborn |
| **환경** | Jupyter Notebook, Google Colab (T4 GPU) |

---

## 📊 분석 파이프라인

### 1단계 — 전통 ML 모델 (`spam_classification_step1.ipynb`)

```
데이터 로드 → 텍스트 전처리 → TF-IDF 벡터화
→ 3개 모델 학습 및 비교 → SHAP 변수 중요도 분석
```

### 2단계 — BERT 파인튜닝 (`spam_classification_step2_bert.ipynb`)

```
BERT Tokenizer → Dataset 구성 → BERT Fine-tuning (3 Epoch)
→ 1단계 대비 성능 비교
```

---

## 📈 모델 성능 비교

| 모델 | Accuracy | AUC |
|------|----------|-----|
| Logistic Regression | 97.13% | 0.9878 |
| Random Forest | 97.49% | 0.9888 |
| XGBoost | 97.22% | 0.9814 |
| **BERT (Fine-tuned)** | **98.92%** | — |

→ BERT 파인튜닝으로 전통 ML 대비 약 **1.4%p 성능 향상** 달성

---

## 🔍 주요 발견

- **스팸 핵심 단어**: `free`, `claim`, `mobile`, `txt`, `stop`, `call` 등 전형적인 스팸 키워드 확인
- **SHAP 분석**: `txt`, `claim`, `ringtone`, `service` 등이 스팸 판단에 높은 영향
- **문자 길이**: 스팸 메시지가 일반 문자보다 평균적으로 길이가 긴 경향
- **BERT 우위**: 문맥을 고려하는 BERT가 단순 단어 빈도 기반 TF-IDF보다 높은 성능

---

## 📁 파일 구조

```
spam-classification/
├── spam_classification_step1.ipynb   # TF-IDF + ML 3개 모델
├── spam_classification_step2_bert.ipynb  # BERT 파인튜닝 (Colab)
├── eda_distribution.png              # 레이블 분포 & 문자 길이 분포
├── model_comparison.png              # 모델 성능 비교 차트
├── confusion_matrix.png              # 혼동행렬 (Random Forest)
├── shap_summary.png                  # SHAP 변수 중요도
└── top_words.png                     # 스팸/햄 주요 단어 비교
```

---

## 🎓 개발 환경

- **로컬**: Python 3.12, Anaconda, Jupyter Notebook
- **클라우드**: Google Colab (Tesla T4 GPU)
