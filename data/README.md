# 데이터

공유 ZIP의 `normal`, `abnormal` 폴더를 `data/raw/` 아래에 배치.

| 경로 | 내용 |
| --- | --- |
| `raw/normal/{1~40}/time_vib/` | 정상 데이터 |
| `raw/abnormal/{1~40}/time_vib/` | 불량 데이터 |

## 구성

| 항목 | 값 |
| --- | --- |
| 정상 CSV | 4,213개 |
| 불량 CSV | 4,187개 |
| 전체 CSV | 8,400개 |
| 파일 형식 | 1,024행 × 2열, 헤더 없음 |
| 열 순서 | 시간, 진동 측정값 |

CSV 읽기: `pd.read_csv(path, header=None, names=["time", "signal"])`

원본은 GitHub 업로드 제외. 데이터 경로는 프로젝트 폴더를 기준으로 지정. 분할 기준은 [프로젝트 README](../README.md) 참고.
