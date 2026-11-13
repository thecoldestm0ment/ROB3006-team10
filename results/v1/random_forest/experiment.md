# 랜덤 포레스트 실험 기록

## 실험 정보

- 실험 ID: `random_forest_stats_v1`
- 샘플 단위: CSV 파일 1개
- 사용 특징: 평균(mean), 표준편차(std), RMS, 최댓값(max), 최솟값(min), 최댓값-최솟값(peak_to_peak)
- 라벨: 정상=0, 불량=1 (양성 클래스)
- 검증: 11~34번 폴더로 학습하고 35~40번 폴더로 검증
- 최종 시험: 11~40번 폴더로 재학습한 뒤 1~10번 폴더를 한 번 평가
- 난수 시드: 42
- Python 라이브러리: scikit-learn 1.7.0, pandas 2.3.0
- 검증 F1 점수로 모델 설정을 선택했으며, 시험 데이터는 설정 선택에 사용하지 않음

## 검증 단계 하이퍼파라미터 비교

| 설정 | F1 | 정밀도(Precision) | 재현율(Recall) |
| --- | ---: | ---: | ---: |
| `{"max_depth": null, "min_samples_leaf": 1, "n_estimators": 200, "random_state": 42}` | 0.5463 | 0.6227 | 0.4865 |
| `{"max_depth": null, "min_samples_leaf": 2, "n_estimators": 200, "random_state": 42}` | 0.5553 | 0.6158 | 0.5055 |
| `{"max_depth": 20, "min_samples_leaf": 1, "n_estimators": 200, "random_state": 42}` | 0.5501 | 0.6250 | 0.4913 |
| `{"max_depth": 20, "min_samples_leaf": 2, "n_estimators": 200, "random_state": 42}` | 0.5540 | 0.6151 | 0.5040 |
| `{"max_depth": null, "min_samples_leaf": 1, "n_estimators": 500, "random_state": 42}` | 0.5409 | 0.6166 | 0.4818 |
| `{"max_depth": null, "min_samples_leaf": 2, "n_estimators": 500, "random_state": 42}` | 0.5490 | 0.6121 | 0.4976 |
| `{"max_depth": 20, "min_samples_leaf": 1, "n_estimators": 500, "random_state": 42}` | 0.5427 | 0.6187 | 0.4834 |
| `{"max_depth": 20, "min_samples_leaf": 2, "n_estimators": 500, "random_state": 42}` | 0.5464 | 0.6106 | 0.4945 |

선택한 설정: `{"max_depth": null, "min_samples_leaf": 2, "n_estimators": 200, "random_state": 42}`

선택 설정의 검증 결과: F1=0.5553, 정밀도=0.6158, 재현율=0.5055

## 최종 시험 결과

- F1: 0.9262
- 정밀도(Precision): 0.9728
- 재현율(Recall): 0.8839
- 혼동행렬: TN=1015, FP=26, FN=122, TP=929
- 혼동행렬 그림: `confusion_matrix.png`

최종 시험 결과는 검증 결과와 분리해 기록했으며, 모델 설정 선택에는 사용하지 않았습니다. 최종 성능을 CSV별 이진 F1로 계산하는 것이 맞는지, 다른 평가 단위나 F1 평균 방식을 사용해야 하는지 교수님께 확인해야 합니다.
