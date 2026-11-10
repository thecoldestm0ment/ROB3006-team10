# ROB3006 · 기계학습론 IC-PBL 10조

모터 진동 데이터를 이용한 정상·불량 분류. 로지스틱 회귀와 랜덤 포레스트의 성능 비교.

## 프로젝트 구조

| 경로 | 내용 |
| --- | --- |
| [data/](data/) | 데이터 안내 및 원본 데이터 저장 위치 |
| [src/preprocess.py](src/preprocess.py) | 공통 데이터 읽기·특징 추출·분할 |
| [src/evaluate.py](src/evaluate.py) | 공통 성능 평가·결과 저장 |
| [notebooks/](notebooks/) | 모델별 실험 Notebook |
| [results/](results/) | 모델별 결과표·그래프·실험 기록 |
| [report/](report/) | 주간보고서·최종보고서 |
| [docs/](docs/) | PPT·발표 대본 |
| [docs/meetings/](docs/meetings/) | 회의록·교수님 피드백 |

## 역할 분담

| 담당 | 작업 |
| --- | --- |
| A · 로지스틱 회귀 | 학습·검증·설정 조정·결과 정리 |
| B · 랜덤 포레스트 | 학습·검증·설정 조정·결과 정리 |
| C · 보고서 | 주간보고서·최종보고서 작성 |
| D · PPT | 데이터 시각화·모델 비교표·발표자료 작성 |
| E · 발표·검증 | 공통 평가·실행 재현·발표 |

공통 전처리는 A·B가 함께 기준을 정하고 관리.

## 데이터 및 실험 기준

원본 ZIP의 `normal`, `abnormal` 폴더를 `data/raw/`에 배치. 원본 데이터는 GitHub 업로드 제외.

CSV 형식: 헤더 없음, 1,024행 × 2열(시간·진동 측정값). 상세 구성은 [데이터 안내](data/README.md) 참고.

| 구분 | 정상·불량 각각의 번호 폴더 |
| --- | --- |
| 내부 학습 | 11~34번 |
| 내부 검증 | 35~40번 |
| 최종 학습 | 11~40번 |
| 최종 시험 | 1~10번 |

- 라벨: 정상 `0`, 불량 `1`
- 기본 특징: 평균, 표준편차, RMS, 최댓값, 최솟값, 최대–최소 차이
- 내부 평가: CSV 단위의 F1(binary, 불량=양성), Precision, Recall, 혼동행렬
- 두 모델에 동일한 특징·분할 적용. CSV를 무작위로 나누지 않고 번호 폴더 단위로 분리
- 로지스틱 회귀 표준화는 학습 데이터로만 계산. 무작위성이 있는 모델·분할은 `random_state=42` 사용
- 모델·설정 확정 후 전체 학습용 데이터로 재학습하고 최종 시험 수행
- 최종 평가 단위·F1 평균 방식은 수업 피드백 기준으로 통일

## 환경 설정

```bash
git clone https://github.com/thecoldestm0ment/ROB3006-team10.git
cd ROB3006-team10
python -m pip install -r requirements.txt
python -m jupyterlab
```

모델별 Notebook: `notebooks/logistic_regression.ipynb`, `notebooks/random_forest.ipynb`

## 실험 기록

각 모델의 `results/` 하위 폴더에 저장.

| 파일 | 내용 |
| --- | --- |
| `metrics.csv` | 실험 ID·평가 지표 |
| `experiment.md` | 특징·분할·설정·실행 방법·결과 해석 |
| `confusion_matrix.png` | 혼동행렬 |

실험 ID 예시: `lr_baseline_01`, `rf_baseline_01`. 보고서·PPT·실험 기록의 수치와 ID 일치.

## 일정

| 주차 | 목표 |
| --- | --- |
| 1주차 | 두 모델 학습·검증·비교, 보고서·PPT 초안 |
| 2주차 | 1차 피드백 반영, 성능 개선, 최종보고서 전체 초안 |
| 3주차 | 2차 피드백 반영, 최종 검증, 보고서 제출·발표 |

금요일 저녁 결과 공유 → 토요일 회의 → 일요일 수정 → 월요일 13:00까지 보고서 제출.

## 협업

작업별 브랜치에서 수정 후 Pull Request로 공유하고, 확인 후 `main`에 병합. 공통 전처리·평가 변경은 두 모델에 함께 반영. PPT·Word 파일은 한 번에 한 명씩 수정.
