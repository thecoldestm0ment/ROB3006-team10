# 진동 분류 v3 — 모터 단위 3개 모델 비교

## 데이터와 평가 방법

- 개발 데이터: 11~40번 폴더 (30개 번호 그룹, 6,308 CSV)
- 같은 번호의 normal/abnormal 폴더는 관계가 불명확해 보수적으로 하나의 그룹으로 묶었습니다.
- 각 모델의 바깥 5-fold에서 모터 폴더를 통째로 보류하고, 안쪽 3-fold에서 특징 세트·설정·판정 기준을 선택했습니다.
- F1 양성 클래스: abnormal=1. 최종 후보 비교 기준은 motor-group nested OOF F1입니다.
- scikit-learn 1.7.0, pandas 2.3.0

## 로지스틱 회귀·랜덤포레스트·RBF SVM 비교

| 모델 | Nested OOF F1 | 정밀도 | 재현율 | Outer fold 평균 F1 ± 표준편차 |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.9172 | 0.9404 | 0.8951 | 0.9130 ± 0.0724 |
| RBF SVM | 0.9067 | 0.9472 | 0.8696 | 0.8984 ± 0.0978 |
| Logistic Regression | 0.8683 | 0.8397 | 0.8989 | 0.8671 ± 0.0546 |

최고 nested F1 모델 계열: **Random Forest** (F1=0.9172).

## 최종 모델 설정

- 모델: Random Forest
- 특징 세트: `v3_amplitude_invariant_16`
- 설정: `{"max_depth": null, "min_samples_leaf": 2, "n_estimators": 200}`
- 기준값: 0.500
- 전체 11~40번 그룹 5-fold 선택용 평균 F1: 0.9454 (표준편차 0.0437)

## 중간 과정 그림

- `figures/motor_spectra_by_class.png`: 모터별 평균 스펙트럼과 클래스 평균·편차
- `figures/representative_waveform_fft.png`: 대표 파형 및 FFT
- `figures/confusion_matrix_*.png`: 모델별 nested OOF 혼동행렬
- `motor_group_metrics.csv`: 각 보류 모터의 모델별 F1

## 사후검증 방지

이전 실험에서 1~10번 테스트 점수를 확인해 개발 방향을 논의했으므로, 이 v3 실행에서는 해당 폴더를 읽지 않았습니다. 과거 점수는 `exploratory_test_1_10.csv`에 탐색 기록으로만 남겼고 최종 시험 점수라고 주장하지 않습니다. 새 독립 모터 데이터가 확보되면 최종 시험에 사용할 수 있습니다.
