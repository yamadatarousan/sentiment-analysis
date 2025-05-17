import streamlit as st
from transformers import pipeline
import plotly.express as px
import pandas as pd

# ページ設定
st.set_page_config(page_title="日本語テキスト分析ダッシュボード", layout="wide")

# タイトルと説明
st.title("日本語テキスト分析ダッシュボード")
st.markdown("""
このダッシュボードでは、入力した日本語テキストの**感情**（ポジティブ/ネガティブ/中立）と**トピック**（例：映画、食事、旅行）を分析します。
テキストを入力し、「分析」ボタンを押してください。結果はグラフで可視化されます。
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

# メインエリア：テキスト入力
st.subheader("テキスト入力")
text = st.text_area(
    "分析する日本語テキストを入力（例：この映画最高！）",
    height=150,
    placeholder="ここにテキストを入力してください"
)

# 分析ボタン
if st.button("分析"):
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

# フッター
st.markdown("---")
st.markdown("Powered by Streamlit, Transformers, and Plotly | Created for AI Text Analysis")