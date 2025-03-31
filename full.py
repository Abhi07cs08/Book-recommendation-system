import streamlit as st
import pandas as pd
import networkx as nx
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from networkx.algorithms import community

# --- Page Configuration and Enhanced Styling ---
st.set_page_config(
    page_title="BookNerd Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a sleek, modern look.
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');
    html, body {
        font-family: 'Montserrat', sans-serif;
        background-color: #f0f2f6;
        color: #333;
    }
    .reportview-container .main .block-container {
        padding: 2rem 2rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #4e54c8, #8f94fb);
        color: white;
        padding: 1rem;
    }
    .stButton > button {
        background-color: #4e54c8;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        font-weight: 600;
    }
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input {
        border-radius: 4px;
        border: 1px solid #ddd;
        padding: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header Banner ---
st.title("📚 BookNerd Recommendation System")
st.write("Discover your next favorite indie book with our modern, interactive recommendation platform.")

# --- Functions for Data, Graphs, and Recommendations ---

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['sentiment'] = df['review'].apply(lambda review: TextBlob(review).sentiment.polarity)
    df['norm_rating'] = df['rating'] / 5.0
    df['norm_sentiment'] = (df['sentiment'] + 1) / 2
    df['edge_weight'] = 0.7 * df['norm_rating'] + 0.3 * df['norm_sentiment']
    aggregated = df.groupby(['user_id', 'book_id'], as_index=False)['edge_weight'].mean()
    return aggregated

def build_bipartite_graph(aggregated):
    B = nx.Graph()
    user_nodes = aggregated['user_id'].unique()
    book_nodes = aggregated['book_id'].unique()
    B.add_nodes_from(user_nodes, bipartite='users')
    B.add_nodes_from(book_nodes, bipartite='books')
    for _, row in aggregated.iterrows():
        B.add_edge(row['user_id'], row['book_id'], weight=row['edge_weight'])
    return B

def compute_similarity(aggregated):
    rating_matrix = aggregated.pivot(index='user_id', columns='book_id', values='edge_weight').fillna(0)
    user_similarity = cosine_similarity(rating_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)
    return rating_matrix, user_similarity_df

def recommend_books(user_id, rating_matrix, user_similarity_df, top_n=5, similar_users_count=3):
    if user_id not in rating_matrix.index:
        return []
    sim_scores = user_similarity_df.loc[user_id].drop(user_id)
    top_similar_users = sim_scores.sort_values(ascending=False).head(similar_users_count)
    target_books = set(rating_matrix.loc[user_id][rating_matrix.loc[user_id] > 0].index)
    rec_scores = {}
    for similar_user, similarity in top_similar_users.items():
        for book, rating in rating_matrix.loc[similar_user].items():
            if rating > 0 and book not in target_books:
                rec_scores[book] = rec_scores.get(book, 0) + similarity * rating
    ranked_recs = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_recs[:top_n]

def plot_most_popular_books(B):
    degrees = B.degree()
    book_degrees = [(node, deg) for node, deg in degrees if B.nodes[node].get('bipartite') == 'books']
    df_books = pd.DataFrame(book_degrees, columns=['book', 'connections'])
    top_books = df_books.sort_values(by='connections', ascending=False).head(10)
    chart = alt.Chart(top_books).mark_bar().encode(
        x=alt.X('book:N', title='Book'),
        y=alt.Y('connections:Q', title='Number of Readers'),
        color=alt.Color('connections:Q', scale=alt.Scale(scheme='blues'))
    ).properties(
        width=600,
        height=400,
        title="Top 10 Most Popular Books"
    )
    st.altair_chart(chart, use_container_width=True)

def plot_recommendations_chart(recommendations, title="Your Recommended Books"):
    if recommendations:
        df_recs = pd.DataFrame(recommendations, columns=["book", "score"])
        chart = alt.Chart(df_recs).mark_bar().encode(
            x=alt.X("book:N", title="Book"),
            y=alt.Y("score:Q", title="Recommendation Score"),
            color=alt.Color("score:Q", scale=alt.Scale(scheme='oranges'))
        ).properties(
            width=600,
            height=400,
            title=title
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("No recommendations available.")

# New functions for additional user-centric stats:
def plot_top_rated_books(file_path):
    # Compute average rating per book from raw data.
    df = pd.read_csv(file_path)
    top_rated = df.groupby('book_id')['rating'].mean().reset_index().sort_values(by='rating', ascending=False).head(10)
    chart = alt.Chart(top_rated).mark_bar().encode(
        x=alt.X('book_id:N', title='Book'),
        y=alt.Y('rating:Q', title='Average Rating'),
        color=alt.Color('rating:Q', scale=alt.Scale(scheme='reds'))
    ).properties(
        width=600,
        height=400,
        title="Top 10 Highest Rated Books"
    )
    st.altair_chart(chart, use_container_width=True)

def plot_rating_distribution(file_path):
    df = pd.read_csv(file_path)
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('rating:Q', bin=alt.Bin(maxbins=20), title='Rating'),
        y=alt.Y('count()', title='Count'),
        color=alt.Color('count()', scale=alt.Scale(scheme='greens'))
    ).properties(
        width=600,
        height=400,
        title="Rating Distribution"
    )
    st.altair_chart(chart, use_container_width=True)

def plot_most_active_users(aggregated):
    user_counts = aggregated.groupby('user_id').size().reset_index(name='count')
    top_users = user_counts.sort_values(by='count', ascending=False).head(10)
    chart = alt.Chart(top_users).mark_bar().encode(
        x=alt.X('user_id:N', title='User'),
        y=alt.Y('count:Q', title='Books Reviewed'),
        color=alt.Color('count:Q', scale=alt.Scale(scheme='purpleblue'))
    ).properties(
        width=600,
        height=400,
        title="Top 10 Most Active Users"
    )
    st.altair_chart(chart, use_container_width=True)

# --- Main Application ---
def main():
    # Sidebar: File Input and Graph Options
    st.sidebar.header("Configuration")
    file_path = st.sidebar.text_input("CSV File Path", "sample_dataset.csv")
    aggregated = load_and_prepare_data(file_path)
    
    st.sidebar.header("Visualizations")
    show_popular_chart = st.sidebar.checkbox("Show Top 10 Popular Books Chart")
    show_top_rated = st.sidebar.checkbox("Show Top 10 Highest Rated Books")
    show_rating_dist = st.sidebar.checkbox("Show Rating Distribution")
    show_active_users = st.sidebar.checkbox("Show Most Active Users")
    show_adv_options = st.sidebar.checkbox("Show Advanced Options (for curious readers)")
    
    B = build_bipartite_graph(aggregated)
    
    if show_popular_chart:
        plot_most_popular_books(B)
    if show_top_rated:
        plot_top_rated_books(file_path)
    if show_rating_dist:
        plot_rating_distribution(file_path)
    if show_active_users:
        plot_most_active_users(aggregated)
    
    if show_adv_options:
        with st.expander("Advanced: Graph Details"):
            st.markdown("**Developer Stats (for the curious):**")
            deg_centrality = nx.degree_centrality(B)
            st.write("Sample Degree Centrality:")
            for node, centrality in list(deg_centrality.items())[:5]:
                st.write(f"{node}: {centrality:.3f}")
            betw_centrality = nx.betweenness_centrality(B, weight='weight')
            st.write("Sample Betweenness Centrality:")
            for node, centrality in list(betw_centrality.items())[:5]:
                st.write(f"{node}: {centrality:.3f}")
    
    # Compute similarity metrics.
    rating_matrix, user_similarity_df = compute_similarity(aggregated)
    
    # Split-screen layout for recommendations.
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Get Recommendations (Existing User)")
        user_id = st.text_input("Enter your User ID", "U003")
        top_n = st.number_input("Number of recommendations", min_value=1, max_value=10, value=5, step=1)
        similar_users_count = st.number_input("Similar users to consider", min_value=1, max_value=10, value=3, step=1)
        if st.button("Get Recommendations for Existing User"):
            recommendations = recommend_books(user_id, rating_matrix, user_similarity_df, top_n=top_n, similar_users_count=similar_users_count)
            if recommendations:
                st.subheader(f"Recommendations for {user_id}:")
                for book, score in recommendations:
                    st.write(f"**{book}** — Score: {score:.2f}")
                plot_recommendations_chart(recommendations, title=f"Recommendations for {user_id}")
            else:
                st.error("User not found or no recommendations available.")
    
    with col2:
        st.header("Custom Profile Recommendations")
        st.write("Select the books you like and get instant recommendations!")
        custom_liked_books = st.multiselect("Select Books You Like", options=sorted(aggregated['book_id'].unique()))
        if st.button("Get Recommendations for Custom Profile"):
            if not custom_liked_books:
                st.error("Please select at least one book.")
            else:
                new_user_profile = pd.DataFrame([{"user_id": "CustomUser", "book_id": book, "edge_weight": 0.85} for book in custom_liked_books])
                aggregated_custom = pd.concat([aggregated, new_user_profile], ignore_index=True)
                rating_matrix_custom, user_similarity_df_custom = compute_similarity(aggregated_custom)
                recommendations_custom = recommend_books("CustomUser", rating_matrix_custom, user_similarity_df_custom)
                if recommendations_custom:
                    st.subheader("Recommendations for Your Custom Profile:")
                    for book, score in recommendations_custom:
                        st.write(f"**{book}** — Score: {score:.2f}")
                    plot_recommendations_chart(recommendations_custom, title="Custom Profile Recommendations")
                else:
                    st.error("No recommendations available for your custom profile.")
    
    st.sidebar.header("Extra Insights")
    if st.sidebar.checkbox("Show Recommendations for All Users"):
        st.write("### Top Recommendations for All Users:")
        for user in rating_matrix.index:
            recs = recommend_books(user, rating_matrix, user_similarity_df, top_n=3)
            st.write(f"**{user}**:")
            for book, score in recs:
                st.write(f"- {book}: {score:.2f}")

if __name__ == "__main__":
    main()
