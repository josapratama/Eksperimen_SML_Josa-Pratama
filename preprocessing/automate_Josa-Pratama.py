"""
Automate Preprocessing - Heart Disease Dataset
Author: Josa Pratama
Description: Script otomatisasi preprocessing dataset Heart Disease yang mengkonversi
             langkah-langkah eksperimen pada notebook menjadi fungsi yang dapat dijalankan
             secara otomatis.
"""

import pandas as pd
import numpy as np
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

RANDOM_SEED = 42
RAW_DATA_PATH = 'heart_disease_raw.csv'
OUTPUT_DIR = 'heart_disease_preprocessing'
DATASET_URL = 'https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv'


def download_dataset(url: str, save_path: str) -> str:
    """
    Mengunduh dataset dari URL dan menyimpannya secara lokal.

    Args:
        url: URL dataset
        save_path: Path lokal untuk menyimpan dataset

    Returns:
        Path file yang berhasil disimpan
    """
    print(f'[INFO] Mengunduh dataset dari: {url}')
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f'[INFO] Dataset berhasil disimpan ke: {save_path}')
    except Exception as e:
        print(f'[WARNING] Gagal mengunduh dataset: {e}')
        print('[INFO] Membuat dataset Heart Disease secara manual...')
        _create_sample_dataset(save_path)
    return save_path


def _create_sample_dataset(save_path: str):
    """Membuat dataset sampel Heart Disease jika download gagal."""
    from sklearn.datasets import make_classification

    np.random.seed(RANDOM_SEED)
    n_samples = 303

    data = {
        'age': np.random.randint(29, 77, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(94, 200, n_samples),
        'chol': np.random.randint(126, 564, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(71, 202, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.round(np.random.uniform(0, 6.2, n_samples), 1),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 4, n_samples),
        'target': np.random.randint(0, 2, n_samples),
    }

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f'[INFO] Dataset sampel berhasil dibuat: {save_path} ({n_samples} baris)')


def load_data(file_path: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV.

    Args:
        file_path: Path file CSV

    Returns:
        DataFrame yang sudah dimuat
    """
    print(f'[INFO] Memuat dataset dari: {file_path}')
    df = pd.read_csv(file_path)

    # Standarisasi nama kolom
    expected_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    if list(df.columns) != expected_cols:
        try:
            df.columns = expected_cols
        except Exception as e:
            print(f'[WARNING] Tidak dapat rename kolom: {e}')

    print(f'[INFO] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom')
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values dengan imputasi median (numerik) dan mode (kategorikal).

    Args:
        df: DataFrame input

    Returns:
        DataFrame tanpa missing values
    """
    print('[INFO] Menangani missing values...')
    df = df.copy()

    missing_before = df.isnull().sum().sum()
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    for col in numerical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f'  [INFO] {col}: diisi dengan median = {median_val}')

    for col in categorical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f'  [INFO] {col}: diisi dengan mode = {mode_val}')

    missing_after = df.isnull().sum().sum()
    print(f'[INFO] Missing values: {missing_before} -> {missing_after}')
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus data duplikat dari dataset.

    Args:
        df: DataFrame input

    Returns:
        DataFrame tanpa duplikat
    """
    print('[INFO] Menghapus data duplikat...')
    before = len(df)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    after = len(df)
    print(f'[INFO] Duplikat dihapus: {before - after} baris. ({before} -> {after})')
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Menghapus outlier menggunakan metode IQR (Interquartile Range).

    Args:
        df: DataFrame input
        columns: List kolom yang akan dibersihkan outliernya

    Returns:
        DataFrame tanpa outlier
    """
    print('[INFO] Menangani outlier dengan metode IQR...')
    df = df.copy()
    before = len(df)

    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        print(f'  [INFO] {col}: hapus {n_before - len(df)} outlier (batas: [{lower:.2f}, {upper:.2f}])')

    df = df.reset_index(drop=True)
    after = len(df)
    print(f'[INFO] Total outlier dihapus: {before - after} baris. ({before} -> {after})')
    return df


def encode_categorical(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Melakukan One-Hot Encoding untuk kolom kategorikal.

    Args:
        df: DataFrame input
        columns: List kolom yang akan di-encode

    Returns:
        DataFrame dengan kolom yang sudah di-encode
    """
    print('[INFO] Melakukan one-hot encoding...')
    before_cols = df.shape[1]

    # Hanya encode kolom yang ada di dataframe
    cols_to_encode = [c for c in columns if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=False, dtype=int)

    after_cols = df.shape[1]
    print(f'[INFO] Kolom: {before_cols} -> {after_cols} (ditambah {after_cols - before_cols} kolom baru)')
    return df


def scale_features(df: pd.DataFrame, columns: list, target_col: str = 'target') -> tuple:
    """
    Standarisasi fitur numerik menggunakan StandardScaler.

    Args:
        df: DataFrame input
        columns: List kolom numerik yang akan distandarisasi
        target_col: Nama kolom target

    Returns:
        Tuple (DataFrame yang sudah distandarisasi, scaler object)
    """
    print('[INFO] Standarisasi fitur numerik...')
    df = df.copy()
    scaler = StandardScaler()

    # Hanya scale kolom yang ada di dataframe
    cols_to_scale = [c for c in columns if c in df.columns]
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    print(f'[INFO] Kolom yang distandarisasi: {cols_to_scale}')
    return df, scaler


def split_data(df: pd.DataFrame, target_col: str = 'target',
               test_size: float = 0.2, random_state: int = RANDOM_SEED) -> tuple:
    """
    Membagi dataset menjadi train dan test set.

    Args:
        df: DataFrame yang sudah dipreproses
        target_col: Nama kolom target
        test_size: Proporsi data test
        random_state: Seed untuk reproduksibilitas

    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    print(f'[INFO] Membagi data (train: {int((1-test_size)*100)}%, test: {int(test_size*100)}%)...')
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f'[INFO] X_train: {X_train.shape}, X_test: {X_test.shape}')
    print(f'[INFO] Distribusi target train:\n{y_train.value_counts().to_dict()}')
    print(f'[INFO] Distribusi target test:\n{y_test.value_counts().to_dict()}')
    return X_train, X_test, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir: str):
    """
    Menyimpan dataset hasil preprocessing ke folder output.

    Args:
        X_train, X_test, y_train, y_test: Data hasil split
        output_dir: Folder output
    """
    os.makedirs(output_dir, exist_ok=True)

    # Simpan train set
    df_train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    train_path = os.path.join(output_dir, 'train.csv')
    df_train.to_csv(train_path, index=False)

    # Simpan test set
    df_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    test_path = os.path.join(output_dir, 'test.csv')
    df_test.to_csv(test_path, index=False)

    # Simpan full preprocessed
    df_full = pd.concat([df_train, df_test], ignore_index=True)
    full_path = os.path.join(output_dir, 'heart_disease_preprocessed.csv')
    df_full.to_csv(full_path, index=False)

    print(f'[INFO] Data tersimpan di: {output_dir}')
    print(f'  - train.csv     : {df_train.shape}')
    print(f'  - test.csv      : {df_test.shape}')
    print(f'  - heart_disease_preprocessed.csv: {df_full.shape}')


def preprocess(
    raw_data_path: str = RAW_DATA_PATH,
    output_dir: str = OUTPUT_DIR,
    download_url: str = DATASET_URL,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED
) -> tuple:
    """
    Fungsi utama untuk menjalankan seluruh pipeline preprocessing secara otomatis.

    Args:
        raw_data_path: Path file data mentah
        output_dir: Folder untuk menyimpan hasil preprocessing
        download_url: URL dataset jika perlu diunduh
        test_size: Proporsi data test
        random_state: Random seed

    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    print('=' * 55)
    print('  AUTOMATE PREPROCESSING - HEART DISEASE DATASET')
    print('  Author: Josa Pratama')
    print('=' * 55)

    # Step 1: Download jika file belum ada
    if not os.path.exists(raw_data_path):
        download_dataset(download_url, raw_data_path)

    # Step 2: Load data
    df = load_data(raw_data_path)

    # Step 3: Handle missing values
    df = handle_missing_values(df)

    # Step 4: Remove duplicates
    df = remove_duplicates(df)

    # Step 5: Remove outliers
    outlier_cols = ['trestbps', 'chol', 'thalach', 'oldpeak']
    df = remove_outliers_iqr(df, outlier_cols)

    # Step 6: One-hot encoding
    multi_cat_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
    df = encode_categorical(df, multi_cat_cols)

    # Step 7: Standarisasi
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df, scaler = scale_features(df, num_cols)

    # Step 8: Split data
    X_train, X_test, y_train, y_test = split_data(df, test_size=test_size, random_state=random_state)

    # Step 9: Simpan hasil
    save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir)

    print('\n' + '=' * 55)
    print('  PREPROCESSING SELESAI!')
    print(f'  Output disimpan di: {output_dir}')
    print('=' * 55)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess()
