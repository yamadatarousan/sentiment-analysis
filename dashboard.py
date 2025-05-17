import streamlit as st
from transformers import pipeline
import plotly.express as px
import pandas as pd
import mysql.connector
from mysql.connector import Error

# ページ設定
st.set_page_config(page_title="日本語テキスト分析ダッシュボード", layout="wide")

# タイトルと説明
st.title("日本語テキスト分析ダッシュボード")
st.markdown("""
このダッシュボードでは、入力した日本語テキストまたはアップロードしたCSVファイルの**感情**（ポジティブ/ネガティブ/中立）と**トピック**（例：映画、食事、旅行）を分析します。
テキストを入力するか、CSVファイルをアップロードして「分析」ボタンを押してください。結果は表とグラフで表示されます。
""")

# モデル初期化（キャッシュで高速化）
@st.cache_resource
def load_models():
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        tokenizer="nlptown/bert-base-multilingual-uncased-sentiment"
    )
    topic_classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        tokenizer="facebook/bart-large-mnli"
    )
    return sentiment_analyzer, topic_classifier

sentiment_analyzer, topic_classifier = load_models()

# サイドバーでトピック設定
st.sidebar.header("トピック設定")
default_topics = ["映画", "食事", "旅行"]
custom_topics = st.sidebar.text_input(
    "カスタムトピック（カンマ区切り、例：仕事,スポーツ）",
    value=""
)
if custom_topics:
    topics = [t.strip() for t in custom_topics.split(",") if t.strip()]
    st.sidebar.write(f"現在のトピック: {topics}")
else:
    topics = default_topics
    st.sidebar.write(f"デフォルトトピック: {topics}")

# MySQL接続設定（後で設定する）
def connect_to_mysql():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",  # MySQLユーザー名
            password="",  # MySQLパスワード
            database="text_analysis_db"
        )
        if connection.is_connected():
            return connection
    except Error as e:
        st.error(f"MySQL接続エラー: {e}")
        return None

# テーブル作成
def create_table():
    connection = connect_to_mysql()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    text TEXT,
                    sentiment VARCHAR(50),
                    topic VARCHAR(50),
                    sentiment_confidence FLOAT,
                    topic_confidence FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.commit()
        except Error as e:
            st.error(f"テーブル作成エラー: {e}")
        finally:
            cursor.close()
            connection.close()

# 分析履歴を保存
def save_to_mysql(text, sentiment, topic, sentiment_confidence, topic_confidence):
    connection = connect_to_mysql()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                INSERT INTO analysis_history (text, sentiment, topic, sentiment_confidence, topic_confidence)
                VALUES (%s, %s, %s, %s, %s)
            """, (text, sentiment, topic, sentiment_confidence, topic_confidence))
            connection.commit()
        except Error as e:
            st.error(f"データ保存エラー: {e}")
        finally:
            cursor.close()
            connection.close()

# メインエリア：テキスト入力とCSVアップロード
st.subheader("入力方法を選択")
input_option = st.radio("入力方法", ["テキスト入力", "CSVアップロード"])

if input_option == "テキスト入力":
    text = st.text_area(
        "分析する日本語テキストを入力（例：この映画最高！）",
        height=150,
        placeholder="ここにテキストを入力してください"
    )
    if st.button("分析", key="text_analyze"):
        if text.strip():
            # 感情分析
            with st.spinner("感情を分析中..."):
                sentiment = sentiment_analyzer(text)
                score = int(sentiment[0]["label"].split()[0])
                sentiment_label = "ポジティブ" if score >= 4 else "ネガティブ" if score <= 2 else "中立"
                sentiment_confidence = sentiment[0]["score"]

            # トピック分類
            with st.spinner("トピックを分析中..."):
                topic_result = topic_classifier(text, candidate_labels=topics, multi_label=False)
                topic_labels = topic_result["labels"]
                topic_scores = topic_result["scores"]

            # MySQLに保存
            create_table()
            save_to_mysql(text, sentiment_label, topic_labels[0], sentiment_confidence, topic_scores[0])

            # 結果表示（2列レイアウト）
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("分析結果")
                st.write(f"**入力テキスト**: {text}")
                st.write(f"**感情**: {sentiment_label} (確信度: {sentiment_confidence:.2%})")
                st.write(f"**トピック**: {topic_labels[0]} (確信度: {topic_scores[0]:.2%})")

            with col2:
                # トピック確信度の棒グラフ
                st.subheader("トピック分類")
                df_topic = pd.DataFrame({"トピック": topic_labels, "確信度": topic_scores})
                fig_bar = px.bar(
                    df_topic,
                    x="トピック",
                    y="確信度",
                    title="トピック分類の確信度",
                    color="トピック",
                    height=300
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # 感情の円グラフ
            st.subheader("感情分析")
            sentiment_df = pd.DataFrame({
                "感情": [sentiment_label],
                "確信度": [sentiment_confidence]
            })
            fig_pie = px.pie(
                sentiment_df,
                names="感情",
                values="確信度",
                title="感情分析の結果",
                height=300
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.error("テキストを入力してください。")

elif input_option == "CSVアップロード":
    uploaded_file = st.file_uploader("CSVファイルをアップロード（テキスト列名：text）", type=["csv"])
    if uploaded_file and st.button("分析", key="csv_analyze"):
        # CSV読み込み
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("CSVファイルに'text'列が必要です。")
        else:
            # バッチ分析
            with st.spinner("CSVデータを分析中..."):
                sentiments = sentiment_analyzer(df["text"].tolist())
                topic_results = topic_classifier(df["text"].tolist(), candidate_labels=topics, multi_label=False)

            # 結果をデータフレームにまとめる
            results = []
            for i, (sent, topic_res) in enumerate(zip(sentiments, topic_results)):
                score = int(sent["label"].split()[0])
                sentiment_label = "ポジティブ" if score >= 4 else "ネガティブ" if score <= 2 else "中立"
                results.append({
                    "テキスト": df["text"].iloc[i],
                    "感情": sentiment_label,
                    "感情確信度": sent["score"],
                    "トピック": topic_res["labels"][0],
                    "トピック確信度": topic_res["scores"][0]
                })
                # MySQLに保存
                create_table()
                save_to_mysql(
                    df["text"].iloc[i],
                    sentiment_label,
                    topic_res["labels"][0],
                    sent["score"],
                    topic_res["scores"][0]
                )

            # 結果表示
            st.subheader("CSV分析結果")
            result_df = pd.DataFrame(results)
            st.dataframe(result_df, use_container_width=True)

            # グラフ：感情分布
            sentiment_counts = result_df["感情"].value_counts().reset_index()
            sentiment_counts.columns = ["感情", "件数"]
            fig_pie = px.pie(
                sentiment_counts,
                names="感情",
                values="件数",
                title="感情分布",
                height=300
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # グラフ：トピック分布
            topic_counts = result_df["トピック"].value_counts().reset_index()
            topic_counts.columns = ["トピック", "件数"]
            fig_bar = px.bar(
                topic_counts,
                x="トピック",
                y="件数",
                title="トピック分布",
                color="トピック",
                height=300
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# フッター
st.markdown("---")
st.markdown("Powered by Streamlit, Transformers, Plotly, and MySQL | Created for AI Text Analysis")