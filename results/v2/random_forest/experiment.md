# 랜덤 포레스트 실험 기록 (v2)

## 실험 조건 및 특징

- 실험 ID: `random_forest_fft_v2`
- 샘플 단위: CSV 파일 1개
- 라벨: 정상=0, 불량=1 (양성 클래스)
- 특징 수: 16개 (시간 영역 6개 + 주파수 영역 10개)
- 주파수 특징: CSV의 시간 간격으로 FFT를 계산하고 DC 성분을 제외했습니다. 나이퀴스트 주파수 기준으로 4개 대역의 파워를 계산했습니다.
- 특징 목록: 평균(mean), 표준편차(std), RMS, 최댓값(max), 최솟값(min), 진폭 범위(peak_to_peak), 최대 파워 주파수(dominant_frequency_hz), 최대 파워 비율(dominant_power_ratio), 스펙트럼 중심(spectral_centroid_hz), 스펙트럼 퍼짐(spectral_spread_hz), 스펙트럼 엔트로피(spectral_entropy), 85% 스펙트럼 롤오프(spectral_rolloff_85_hz), 나이퀴스트 0~10% 대역 파워, 나이퀴스트 10~25% 대역 파워, 나이퀴스트 25~50% 대역 파워, 나이퀴스트 50~100% 대역 파워
- 그룹 분리: 같은 번호 폴더의 CSV가 서로 다른 교차검증 fold에 나뉘지 않도록 했습니다.
- 난수 시드: 42
- Python 라이브러리: scikit-learn 1.7.0, pandas 2.3.0

## 학습 데이터(11~34번) 그룹 교차검증

폴더 번호를 기준으로 4개 그룹 fold를 구성했습니다. fold 평균 F1이 가장 높은 설정을 선택하고, 동률이면 fold별 F1 편차가 낮은 설정을 우선했습니다.

| 설정 | 교차검증 평균 F1 | F1 표준편차 | 평균 정밀도 | 평균 재현율 |
| --- | ---: | ---: | ---: | ---: |
| `{"max_depth": null, "min_samples_leaf": 1, "n_estimators": 200}` | 0.9774 | 0.0337 | 0.9626 | 0.9952 |
| `{"max_depth": null, "min_samples_leaf": 2, "n_estimators": 200}` | 0.9765 | 0.0333 | 0.9626 | 0.9935 |
| `{"max_depth": 20, "min_samples_leaf": 1, "n_estimators": 200}` | 0.9774 | 0.0337 | 0.9626 | 0.9952 |
| `{"max_depth": 20, "min_samples_leaf": 2, "n_estimators": 200}` | 0.9765 | 0.0333 | 0.9626 | 0.9935 |
| `{"max_depth": null, "min_samples_leaf": 1, "n_estimators": 500}` | 0.9757 | 0.0329 | 0.9626 | 0.9919 |
| `{"max_depth": null, "min_samples_leaf": 2, "n_estimators": 500}` | 0.9757 | 0.0329 | 0.9626 | 0.9919 |
| `{"max_depth": 20, "min_samples_leaf": 1, "n_estimators": 500}` | 0.9757 | 0.0329 | 0.9626 | 0.9919 |
| `{"max_depth": 20, "min_samples_leaf": 2, "n_estimators": 500}` | 0.9757 | 0.0329 | 0.9626 | 0.9919 |

선택한 설정: `{"max_depth": null, "min_samples_leaf": 1, "n_estimators": 200}`

- fold 평균 F1: 0.9774 (표준편차 0.0337)
- 교차검증 전체 예측(OOF) F1: 0.9761

## 별도 검증 및 최종 테스트

- 검증: 11~34번 폴더로 학습하고 35~40번 폴더로 평가
- 최종 테스트: 11~40번 폴더로 재학습하고 1~10번 폴더로 한 번 평가
- 검증 F1: 0.7747, 정밀도=0.6323, 재현율=1.0000
- 최종 테스트 F1: 0.9368, 정밀도=1.0000, 재현율=0.8811
- 최종 테스트 혼동행렬: TN=1041, FP=0, FN=125, TP=926
- 폴더별 검증·테스트 점수: `folder_metrics.csv`
- 최종 테스트 혼동행렬 그림: `confusion_matrix.png`

전체 F1만으로 일반화 성능을 판단하지 말고 `folder_metrics.csv`의 폴더별 결과도 함께 확인해야 합니다. 교차검증과 35~40번 검증 결과는 최종 테스트 데이터와 분리했습니다. 최종 평가 단위와 F1 평균 방식은 수업 피드백에 따라 확인해야 합니다.
