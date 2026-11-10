"""공통 전처리 뼈대. 모델 담당 두 명이 기준을 맞춘 뒤 구현합니다.

구현 순서:
1. normal/abnormal 안의 번호 폴더와 CSV 목록 읽기
2. CSV를 header=None으로 읽고 시간·측정값 구분
3. 파일별 공통 특징 추출
4. 번호 폴더 단위로 내부 학습·검증 분리

시험용 1~10번은 최종 평가 전 모델 선택에 사용하지 않습니다.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

LABELS = {"normal": 0, "abnormal": 1}
TRAIN_IDS = tuple(range(11, 35))
VALIDATION_IDS = tuple(range(35, 41))
FINAL_TRAIN_IDS = tuple(range(11, 41))
TEST_IDS = tuple(range(1, 11))
RANDOM_STATE = 42

FEATURE_NAMES = (
    "mean",
    "std",
    "rms",
    "max",
    "min",
    "peak_to_peak",
)


def extract_features(signal):
    """한 CSV 파일의 측정값에서 FEATURE_NAMES 순서의 특징을 반환합니다."""
    # TODO: 표준편차의 정의 등 세부 계산 기준도 두 모델에서 통일합니다.
    raise NotImplementedError("공통 특징 추출 처리를 구현해주세요.")


def load_split(split, data_root=DATA_ROOT):
    """지정한 분할의 특징·라벨·원본 파일 정보를 반환할 예정입니다.

    split: train / validation / final_train / test
    반환 형식은 모델 담당자와 검증 담당자가 맞춘 뒤 구현합니다.
    원본 정보에는 class_name, folder_id, 원본 경로를 남깁니다.
    """
    # TODO: CSV를 무작위로 섞어 나누지 않고 번호 폴더 단위로 분리합니다.
    raise NotImplementedError("데이터 읽기·분할 처리를 구현해주세요.")
