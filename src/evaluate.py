"""공통 평가 뼈대. 검증 담당자가 모델 담당자와 형식을 맞춘 뒤 구현합니다.

기본 기준: 정상=0, 불량=1, binary F1 / Precision / Recall.
혼동행렬의 행·열 순서는 [0, 1]로 고정하고 그림에 라벨을 표시합니다.
최종 평가 단위와 F1 평균 방식은 교수님 피드백을 반영합니다.
"""


def evaluate_predictions(y_true, y_pred):
    """정답과 예측값으로 지표와 혼동행렬을 반환할 예정입니다."""
    # TODO: F1·Precision·Recall과 zero_division 처리 기준을 정합니다.
    raise NotImplementedError("공통 평가 처리를 구현해주세요.")


def save_results(result, output_dir, experiment_id):
    """실험 ID와 함께 결과표·혼동행렬을 저장할 예정입니다."""
    # TODO: metrics.csv 열 이름과 experiment.md의 실험 ID를 맞춥니다.
    raise NotImplementedError("결과 저장 처리를 구현해주세요.")
