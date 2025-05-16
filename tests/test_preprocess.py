from preprocess import load_data, normalize_data

def test_load_data():
    df = load_data()
    assert not df.empty

def test_normalize_data():
    df = load_data()
    df_norm = normalize_data(df)
    assert abs(df_norm.iloc[:, 0].mean()) < 1
