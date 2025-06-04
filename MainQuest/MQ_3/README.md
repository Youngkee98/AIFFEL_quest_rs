# 📘 LSTM 기반 텍스트 분류 모델에서의 Vocab Size와 클래스 불균형의 영향 분석

이 저장소는 다음 논문의 코드, 데이터 설정, 실험 결과를 포함하고 있습니다:

**"LSTM 기반 텍스트 분류 모델에서의 Vocab Size와 클래스 불균형의 영향 분석"**

## 📌 개요

본 연구는 Reuters-21578 뉴스 데이터셋을 기반으로, 단어 집합 크기(vocabulary size)와 클래스 불균형이 LSTM 계열 모델(LSTM, BiLSTM, GRU)의 성능에 미치는 영향을 실증적으로 분석합니다.

핵심 내용은 다음과 같습니다:
- 전체 단어 분포의 누적 빈도를 기반으로 vocab size를 10%~100% 범위로 조정
- 클래스 불균형 대응을 위한 가중치 손실 함수 및 데이터 증강 기법 평가
- Accuracy 및 macro F1-score를 주요 성능 지표로 사용
- F1-score 기반 Stability 지표를 활용한 vocab size별 모델 안정성 분석

---

## 🧠 실험 모델

- LSTM (Long Short-Term Memory) [1]
- BiLSTM (Bidirectional LSTM) [2]
- GRU (Gated Recurrent Unit) [3]

모든 모델은 아래와 같은 공통 아키텍처를 따릅니다:
- Embedding → RNN 계층 (64, 32 유닛) → Dropout → Dense (64) → Softmax 출력 (46개 클래스)

---

## 📈 핵심 결과 요약
![F1_Comparison](https://github.com/user-attachments/assets/4ce0ee58-bac2-4163-a250-820866cf1f42)
![Accuracy_Comparison](https://github.com/user-attachments/assets/f93bec66-3965-4849-920f-2bfc4281c711)



- BiLSTM 모델이 정확도 및 F1 점수 모두에서 가장 우수한 성능을 보임
- Vocab Size가 436 이상(43.6%)일 때 BiLSTM은 가장 안정적인 성능을 유지
- 데이터 증강이 클래스 불균형 문제 완화에 효과적
- Vocab size 선택 전략은 과적합과 정보 손실을 모두 방지하는 데 중요함

---

## 📬 문의

**김영기 (Youngkee Kim)**  
zerokee98@naver.com  
AI Research 13th, MODU LABS

---

# 📘 Effect of Vocabulary Size and Class Imbalance on LSTM-based Text Classification Models

This repository contains the code, data settings, and experimental results for our paper:

**"Effect of Vocabulary Size and Class Imbalance on LSTM-based Text Classification Models"**

## 📌 Overview

This study explores how vocabulary size and class imbalance influence the performance of RNN-based models—specifically LSTM, BiLSTM, and GRU—in multi-class text classification tasks using the Reuters-21578 dataset.

Key aspects include:
- Systematic variation of vocabulary size based on cumulative token coverage (from 10% to 100%)
- Evaluation of class imbalance handling methods: weighted loss and data augmentation
- Use of accuracy and macro F1-score as primary evaluation metrics
- Stability analysis across vocab sizes using the F1-based stability index

---

## 🧠 Models Evaluated

- Long Short-Term Memory (LSTM) [1]
- Bidirectional LSTM (BiLSTM) [2]
- Gated Recurrent Unit (GRU) [3]

All models share a common architecture:
- Embedding → RNN Layers (64, 32 units) → Dropout → Dense (64) → Output (Softmax over 46 classes)

---

## 📈 Key Findings

- BiLSTM outperforms LSTM and GRU in both accuracy and F1
- From vocab size 436 (43.6% coverage) and above, BiLSTM shows robust and stable performance
- Data augmentation effectively mitigates the impact of class imbalance
- Vocabulary size selection plays a key role in performance and generalization

---

## 📬 Contact

**Youngkee Kim**  
zerokee98@naver.com  
AI Research 13th, MODU LABS
