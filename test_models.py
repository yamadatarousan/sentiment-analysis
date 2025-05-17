from transformers import pipeline

# 感情分析モデル
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
text = "この映画最高！"
sentiment = sentiment_analyzer(text)
score = int(sentiment[0]["label"].split()[0])
label = "ポジティブ" if score >= 4 else "ネガティブ" if score <= 2 else "中立"
print(f"感情: {label} (確信度: {sentiment[0]['score']:.2%})")

# トピック分類モデル
topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
topics = ["映画", "食事", "旅行"]
topic_result = topic_classifier(text, candidate_labels=topics)
print(f"トピック: {topic_result['labels'][0]} (確信度: {topic_result['scores'][0]:.2%})")