# v4 (3주차) — 특징 추출 방법 비교 (랜덤포레스트 설정 고정)

## 목적

2주차 v3 최종 랜덤포레스트를 기준으로, 특징 추출 방법만 바꿨을 때 처음 보는 모터 번호에 대한 불량 탐지 F1이 개선되는지 확인한다. 설정을 고정한 특징 비교이므로, v3 보고서의 중첩 교차검증 F1(0.9172)과 직접 비교하지 않고 같은 조건에서 다시 실행한 A를 기준으로 개선 폭을 계산한다.

## 데이터와 평가 조건

- 개발 데이터: 정상·불량 11~40번 폴더만 사용 (정상 3,172 + 불량 3,136 = 6,308 CSV, 모두 1,024 샘플). **1~10번은 읽지 않았다.**
- CSV 한 개 = 예측 한 건, 불량(abnormal) = 양성 클래스.
- 그룹: 모터 번호. 정상·불량 폴더의 같은 번호는 같은 그룹.
- 분할 세트 3개 (모두 모터 번호 단위 5-fold, 각 fold 6개 번호). 각 세트 안에서 A·B·C는 같은 학습·검증 번호를 사용.
  - `v3_outer_folds`: `results/v3/nested_cv_folds.csv`에 기록된 v3 바깥 5-fold를 그대로 복원. 세 모델의 기록이 같음을 확인함.
  - `motor_shuffle_seed42`, `motor_shuffle_seed73`: `numpy.random.default_rng(seed).permutation(11..40)`을 5등분.
  - 전체 목록: `split_definitions.csv`
- 주 지표: 각 분할 세트에서 검증 예측을 모두 합친 OOF F1. Precision, Recall, FP, FN도 함께 기록.

## 특징

| 방법 | 개수 | 정의 |
| --- | ---: | --- |
| A. v3 기준선 | 16 | `preprocess_v3.AMPLITUDE_INVARIANT_FEATURES_V3`. 기존 `extract_features_v2` + `extract_shape_features_v3` 사용 |
| B. Welch 16대역 | 16 | 평균 제거 → `scipy.signal.welch`(Hann, nperseg 512, noverlap 256, detrend 없음, density) → DC 제외 양의 주파수를 동일 폭 16대역 → 대역 전력 / 전체 전력 → log10 |
| C. sym8 웨이블릿 패킷 | 32 | 평균 제거 → RMS 정규화 → `pywt.WaveletPacket(sym8, periodization, maxlevel=5)` → 5단계 노드 32개를 `order="freq"`로 정렬 → 노드 계수 제곱합 / 32개 노드 합 |

- **샘플링 주파수 근거:** v2와 같이 각 CSV의 시간 열 간격 중앙값으로 계산했다(`preprocess_v2.extract_features_v2`). 측정값은 400.0~401.6 Hz이고 중앙값은 401.6 Hz이다. 대역은 Nyquist 비율로 정의했으므로 파일별 fs 차이와 상관없이 같은 FFT 빈 범위를 쓴다.
- **B 대역 경계:** j번째 대역(j=1..16)은 ((j−1)/16·Nyquist, j/16·Nyquist]이다. DC는 제외하고 Nyquist 빈은 16번 대역에 넣었다. 1,024 샘플·nperseg 512 기준으로 대역마다 PSD 빈이 정확히 16개씩이다. fs 401.6 Hz 기준 대역 폭은 약 12.55 Hz(Nyquist 200.8 Hz)이다.
- **B의 0 전력 처리:** 전체 전력이 0이면 16대역을 모두 1/16로 둔다. 상대 전력이 1e−12 미만이면 1e−12로 잘라 log10을 적용한다. 이번 데이터에서 두 경우 모두 0건이었다.
- **C의 0 신호 처리:** RMS가 0이면 32노드를 모두 1/32로 둔다. 이번 데이터에서 0건이었다.
- **C의 노드 의미:** 노드 i는 대략 ((i−1)/32·Nyquist, i/32·Nyquist] 대역에 해당한다(필터 중첩으로 경계는 근사). 47·110·160 Hz 사인파가 각각 8·18·26번 노드에 최대 에너지를 갖는 것으로 순서를 확인했다.

## 모델 (세 방법 공통, 고정)

`RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1)`. 나머지는 v3 `_make_model`과 같이 scikit-learn 기본값이다. 불량 판정은 `predict_proba ≥ 0.5`이고, v3 `_predict`와 같은 규칙이다.

## 결과

### 분할별 OOF F1

| 방법 | v3 기존 분할 | seed 42 | seed 73 | 평균 | 최저 | 평균 차이(vs A) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A. v3 기준선 | 0.9464 | 0.9589 | 0.9318 | 0.9457 | 0.9318 | — |
| B. Welch 16대역 | **0.9896** | **0.9955** | **0.9974** | **0.9942** | **0.9896** | +0.0485 |
| C. sym8 WPT 32 | 0.9861 | 0.9931 | 0.9944 | 0.9912 | 0.9861 | +0.0455 |

세 분할 모두 순위가 B > C > A로 같다.

### v3 기존 분할의 세부 지표

| 방법 | F1 | Precision | Recall | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.9464 | 0.9646 | 0.9289 | 107 | 223 |
| B | 0.9896 | 0.9968 | 0.9825 | 10 | 55 |
| C | 0.9861 | 0.9987 | 0.9739 | 4 | 82 |

### 모터별 오류 (v3 기존 분할, `motor_errors_v3_outer_folds.csv`)

| 모터 | 불량 CSV | FN A → B / C | 정상 CSV | FP A → B / C |
| ---: | ---: | --- | ---: | --- |
| 24 | 103 | 103 → 54 / 49 | 100 | 0 → 0 / 0 |
| 18 | 102 | 88 → 1 / 32 | 105 | 0 → 0 / 0 |
| 36 | 104 | 19 → 0 / 0 | 104 | 104 → 0 / 0 |
| 34 | 103 | 10 → 0 / 0 | 102 | 0 → 0 / 0 |
| 16 | 103 | 2 → 0 / 0 | 103 | 0 → 0 / 0 |
| 26 | 107 | 1 → 0 / 0 | 104 | 0 → 0 / 0 |

FP가 늘어난 곳은 B가 모터 12(0→5)와 14(0→5), C가 모터 12(0→2)와 14(0→2)이다. C는 모터 37에서 FN이 1건 생겼다(0→1).

### 실행 시간 (Python 3.14.7, 6,308 CSV)

| 항목 | A | B | C |
| --- | ---: | ---: | ---: |
| 특징 추출 (전체 CSV) | 2.98 s | 3.71 s | 3.35 s |
| 학습·예측 (3개 분할 × 5 fold) | 13.48 s | 11.83 s | 16.17 s |

CSV 읽기는 세 방법이 공유하며 12.0 s 걸렸다. 전체 실행 시간은 66.6 s이다.

## 해석과 한계

- 평균 F1이 가장 높은 B(Welch 16대역)를 다음 단계의 1순위 후보로 둔다. C와의 차이(0.003)는 작다.
- 랜덤포레스트 설정과 임계값 0.5는 v3가 A에 맞춰 고른 값이다. 특히 `v3_outer_folds`는 A의 설정을 고른 바로 그 분할이라 A에 유리한 조건인데도 B와 C가 앞섰다.
- 모터가 30개뿐이라 모터 한두 개의 판정이 F1을 크게 움직인다. A와 B의 차이도 대부분 모터 18, 24, 36에서 나온다.
- B와 C는 차원(16 대 32)과 표현이 함께 달라서, 차이를 어느 쪽 효과로 볼지 분리되지 않는다.
- 점수가 0.99 수준이다. 이것이 모터 상태 차이를 반영하는지, 측정 조건 차이를 학습한 것인지는 아직 확인하지 않았다.

## 후속 실험 (이번에 실행하지 않음)

- A+B, A+C 결합 특징 비교
- 임계값·하이퍼파라미터를 안쪽 교차검증으로 고르는 중첩 교차검증으로 B 재평가
- 반복 분할 확대와 모터 단위 부트스트랩으로 ΔF1의 신뢰구간 계산
- 모터 24의 누락 원인 분석
- 최종 설정 확정 후 11~40번 전체 학습 → 1~10번 1회 평가 코드 작성

## 재현

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m src.run_experiments_v4 --data-root "<normal/과 abnormal/이 들어 있는 폴더>"
```

- 버전: numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, scipy 1.18.1, PyWavelets 1.8.0. v3는 scikit-learn 1.7.0으로 실행되었다.
- 산출물: `summary.csv`, `oof_metrics.csv`, `fold_metrics.csv`, `split_definitions.csv`, `motor_errors_v3_outer_folds.csv`, `timing.csv`, `run_info.json`, `oof_predictions.csv.gz`, `feature_cache.csv.gz`(CSV별 A·B·C 특징)
