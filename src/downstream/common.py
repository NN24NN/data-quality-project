"""Tiền xử lý và chia train/test dùng chung cho 3 tác vụ downstream."""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def with_preprocessing(model):
    """Điền ô thiếu bằng trung vị rồi chuẩn hóa z-score trước khi vào mô hình (đều học trên tập train).

    Nhóm one-hot bị làm thiếu (cả nhóm = 0) giữ nguyên: mô hình thấy đó là "không rõ giá trị".
    """
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), model)


def xy(df, target):
    """Tách (X, y) thành mảng numpy — không dựa vào index, vì dữ liệu nhân bản có index trùng."""
    return df.drop(columns=target).to_numpy(), df[target].to_numpy()


def fit_and_score(model, train, test, target, metric):
    X_train, y_train = xy(train, target)
    X_test, y_test = xy(test, target)
    model = with_preprocessing(model).fit(X_train, y_train)
    return float(metric(y_test, model.predict(X_test)))


def split_train_test(df, target, target_type, test_size, seed, n_bins=10):
    """Chia stratified theo target; target số được rời rạc hóa thành n_bins khoảng theo phân vị."""
    strata = df[target]
    if target_type == 'numeric':
        strata = pd.qcut(strata, n_bins, labels=False, duplicates='drop')
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=strata)
