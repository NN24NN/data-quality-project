"""3 tác vụ downstream, cùng chữ ký: evaluate(algorithm, train, test, target, seed, params) -> điểm hiệu năng.

Khóa của EVALUATORS trùng tên task trong configs/experiment_matrix.yaml;
`params` là siêu tham số của thuật toán, nơi gọi đọc từ configs/algorithms.yaml.
"""
from . import classification, clustering, regression

EVALUATORS = {
    'classification': classification.evaluate,
    'regression': regression.evaluate,
    'clustering': clustering.evaluate,
}
