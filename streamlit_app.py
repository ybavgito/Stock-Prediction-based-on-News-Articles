import streamlit as st
from name_ticker_module import find_similar_names_and_tickers
from bing_search_module import bing_search_news
from polygon_module import get_stock_price_history
from calendar_module import get_first_last_days
from cohere_rerank_module import rerank_search_results
from financial_bert_module import predict_monthly_stock_price_change_rate
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Predictor", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0072ff;
        text-align: center;
        padding-bottom: 10px;
    }
    .metric-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #262730;
    }
    .highlight-box {
        background-color: #f1f3f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='main-title'>📈 Stock Analysis and Prediction</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🔎 Select Company")
    user_input = st.text_input("Enter company name", "Tesla")

# Main layout
if user_input:
    search_results = find_similar_names_and_tickers(user_input)
    option = st.selectbox('Choose a company:', search_results)
    name, ticker = option

    if st.button('🚀 Analyze'):
        with st.spinner("Fetching news and stock data..."):
            # Step 1: Fetch and rerank articles
            this_month_articles = bing_search_news(name)
            articles_content = [article['description'] for article in this_month_articles]
            reranked_articles_content = rerank_search_results(name, articles_content, 5)

            # Step 2: Predict price change
            concatenated_articles_content = " ".join(reranked_articles_content)
            monthly_stock_price_change_rate = predict_monthly_stock_price_change_rate(concatenated_articles_content)

            # Step 3: Get historical price and calculate prediction
            past_month_first_day_str, current_month_last_day_str, year_month_labels = get_first_last_days()
            stock_price_history = get_stock_price_history(ticker, past_month_first_day_str, current_month_last_day_str)
            predicted_next_month_stock_price = stock_price_history[-1] * (1 + monthly_stock_price_change_rate)
            stock_price_history.append(predicted_next_month_stock_price)

        # Display metrics
        col1, col2 = st.columns(2)
        col1.metric("📊 Predicted Change Rate", f"{monthly_stock_price_change_rate * 100:.2f}%")
        col2.metric("💰 Next Month Price", f"${predicted_next_month_stock_price:.2f}")

        # Show articles in expander
        with st.expander("📰 Top Financial News"):
            for idx, article in enumerate(reranked_articles_content):
                st.markdown(f"- {article}")

        # Plotting in compact container
        st.markdown("### 📉 Stock Price Trend")
        with st.container():
            fig, ax = plt.subplots(figsize=(8, 4))  # Reduced size
            ax.plot(year_month_labels[:-1], stock_price_history[:-1], label='History', marker='o')
            ax.plot(year_month_labels[-2:], stock_price_history[-2:], label='Prediction', color='red', marker='o')
            ax.set_xlabel('Year-Month')
            ax.set_ylabel('Stock Price')
            ax.set_title(f'{name} Stock Price History and Prediction', fontsize=12)
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
