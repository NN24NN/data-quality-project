"""5 polluter độc lập, cùng chữ ký: pollute(df, level, seed, target, target_type, onehot_prefixes) -> df.

Khóa của POLLUTERS trùng tên dimension trong configs/experiment_matrix.yaml.
Mọi polluter giữ index của dòng gốc và không sửa df đầu vào.
"""
from . import class_balance, completeness, feature_accuracy, target_accuracy, uniqueness

POLLUTERS = {
    'completeness': completeness.pollute,
    'feature_accuracy': feature_accuracy.pollute,
    'target_accuracy': target_accuracy.pollute,
    'uniqueness': uniqueness.pollute,
    'class_balance': class_balance.pollute,
}
