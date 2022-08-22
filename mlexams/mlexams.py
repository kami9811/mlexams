# model libraries
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score
import numpy as np
from typing import Dict, Union, List

def get_accuracy(
  train_data: any,
  train_label: any,
  test_data: any,
  test_label: any,
  label_kind: int,
  model_kind: str,
  options: Dict[str, Union[int, str]] = {
    "C": 5, "kernel": 'rbf', "gamma": 'auto'
  }
) -> float:


    if model_kind == "scv":
        # learning
        # clf = SVC(C=1, kernel='rbf', gamma='auto')
        clf = SVC(**options)
        clf.fit(train_data, train_label)

        p = clf.predict(test_data)
        accuracy = accuracy_score(
            test_label,
            p
        )

    if model_kind == "krr":
        # one-hot coding
        label_kind = label_kind
        train_label = (
            (
                -1 * np.ones((label_kind, label_kind), dtype=int)
            ) + (
                2 * np.eye(label_kind)
            )
        )[train_label]
        test_label = (
            (
                -1 * np.ones((label_kind, label_kind), dtype=int)
            ) + (
                2 * np.eye(label_kind)
            )
        )[test_label]

        # learning
        clf = KernelRidge(alpha=0.2, kernel='rbf')
        clf.fit(train_data, train_label)

        p = clf.predict(test_data)
        accuracy = accuracy_score(
            np.argmax(test_label, axis=1), 
            np.argmax(p, axis=1)
        )

    return accuracy