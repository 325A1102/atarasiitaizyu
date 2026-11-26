import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="スマートフィットネス予測", page_icon="⌚", layout="wide")

@st.cache_resource
def train_model():
    # 少し本格的なダミーデータを300件生成（年齢、性別、体重、心拍数、運動時間）
    np.random.seed(42)
    n = 300
    age = np.random.randint(20, 60, n)
    gender = np.random.randint(0, 2, n) # 0: 女性, 1: 男性
    weight = np.random.uniform(50, 100, n)
    heart_rate = np.random.randint(100, 180, n)
    duration = np.random.randint(10, 120, n)

    # 回帰分析用の目的変数（消費カロリー）を計算（※架空の計算式にノイズを追加）
    calories = (age * 0.5) + (gender * 40) + (weight * 1.5) + (heart_rate * 2.5) + (duration * 6.0) - 200
    calories += np.random.normal(0, 30, n) # ばらつき（ノイズ）を追加

    df = pd.DataFrame({'age': age, 'gender': gender, 'weight': weight, 'heart_rate': heart_rate, 'duration': duration, 'calories': calories})

    X = df[['age', 'gender', 'weight', 'heart_rate', 'duration']]
    y = df['calories']

    # 線形回帰モデルの学習
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model()

st.title("⌚ スマートウォッチ風 消費カロリー予測AI")
st.markdown("年齢や平均心拍数などのデータを組み合わせて、多重回帰分析でパーソナライズされたカロリー予測を行います。")

# 入力フォームを左右のカラムで整理
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 あなたのプロフィール")
    age = st.number_input("年齢", min_value=10, max_value=100, value=30)
    gender_str = st.radio("性別", ["女性", "男性"], horizontal=True)
    gender = 1 if gender_str == "男性" else 0
    weight = st.number_input("体重 (kg)", min_value=30.0, max_value=150.0, value=60.0, step=0.5)

with col2:
    st.subheader("🏃‍♂️ 運動のデータ")
    heart_rate = st.slider("平均心拍数 (bpm)", min_value=80, max_value=200, value=130)
    duration = st.slider("運動時間 (分)", min_value=5, max_value=180, value=45)

st.write("---")

if st.button("AIでカロリーを計算する", type="primary", use_container_width=True):
    # 予測の実行
    input_df = pd.DataFrame([[age, gender, weight, heart_rate, duration]],
                            columns=['age', 'gender', 'weight', 'heart_rate', 'duration'])
    pred_cal = model.predict(input_df)[0]
    pred_cal = max(0, pred_cal) # マイナスになるのを防止

    st.success("計算が完了しました！")

    # 結果表示エリア（左に数値、右にグラフ）
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(label="🔥 予測消費カロリー", value=f"{pred_cal:.0f} kcal", delta="回帰分析による算出")

    # 追加機能：運動時間ごとの消費カロリー推移シミュレーショングラフ
    with res_col2:
        st.markdown("##### 📈 運動時間と消費カロリーのシミュレーション")
        durations = range(10, 121, 10) # 10分〜120分まで10分刻み
        sim_data = []
        for d in durations:
            sim_df = pd.DataFrame([[age, gender, weight, heart_rate, d]],
                                  columns=['age', 'gender', 'weight', 'heart_rate', 'duration'])
            sim_cal = max(0, model.predict(sim_df)[0])
            sim_data.append({"時間 (分)": d, "カロリー (kcal)": sim_cal})

        # グラフの描画
        st.line_chart(pd.DataFrame(sim_data).set_index("時間 (分)"))
