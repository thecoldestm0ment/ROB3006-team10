"""Train the fixed model on folders 11–40 and print CSV-level test metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data_loader import TEST_IDS, TRAIN_IDS, load_dataset
from .metrics import classification_metrics
from .model_config import THRESHOLD, make_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="folder containing normal/ and abnormal/",
    )
    args = parser.parse_args()
    root = args.data_root.expanduser()

    X_train, y_train = load_dataset(root, TRAIN_IDS)
    print(f"학습 데이터: 11~40번, CSV {len(y_train):,}개", flush=True)
    model = make_model()
    model.fit(X_train, y_train)

    # Read the held-out test folders only after training has finished.
    X_test, y_test = load_dataset(root, TEST_IDS)
    probabilities = model.predict_proba(X_test)[:, 1]
    y_pred = (probabilities >= THRESHOLD).astype(np.int8)
    scores = classification_metrics(y_test, y_pred)

    print(f"시험 데이터: 1~10번, CSV {len(y_test):,}개")
    print("모델: Random Forest, Welch 32대역, random_state=42")
    print(f"시험 F1 (CSV 단위): {scores['f1']:.4f}")
    print(f"정밀도: {scores['precision']:.4f}")
    print(f"재현율: {scores['recall']:.4f}")
    print("혼동행렬 [[TN, FP], [FN, TP]]:")
    print(f"[[{scores['tn']}, {scores['fp']}], [{scores['fn']}, {scores['tp']}]]")
    print("참고: 1~10번은 프로젝트 이전 실험에서도 사용한 기존 시험 데이터입니다.")


if __name__ == "__main__":
    main()
