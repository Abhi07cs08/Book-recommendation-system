import streamlit as st
import pandas as pd
import networkx as nx
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from networkx.algorithms import community
import matplotlib.pyplot as plt

# --- Page Configuration and Custom Styling ---
st.set_page_config(
    page_title="BookNerd Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for a modern look.
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

st.title("📚 BookNerd Recommendation System")
st.write("Discover your next favorite indie book with our modern recommendation platform.")

# --- Functions for Data Loading, Graph Building, and Recommendations ---

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

def plot_bipartite_graph(B):
    users = [n for n, d in B.nodes(data=True) if d['bipartite'] == 'users']
    pos = nx.bipartite_layout(B, users)
    plt.figure(figsize=(8, 6))
    nx.draw(B, pos, with_labels=True, node_size=300, font_size=8)
    plt.title("Bipartite Graph: Users and Books")
    st.pyplot(plt)

# --- Main Application ---

def main():
    # File path input.
    file_path = st.text_input("Enter path to CSV file", "sample_dataset.csv")
    aggregated = load_and_prepare_data(file_path)
    
    # Sidebar: Graph Analysis Options.
    st.sidebar.header("Graph Analysis Options")
    show_edges = st.sidebar.checkbox("Show Graph Edges")
    show_metrics = st.sidebar.checkbox("Show Graph Metrics")
    show_popular = st.sidebar.checkbox("Show Most Popular Books")
    show_influencers = st.sidebar.checkbox("Show Top Influencers")
    plot_graph_flag = st.sidebar.checkbox("Plot Graph")
    
    B = build_bipartite_graph(aggregated)
    if show_edges:
        st.markdown("**Graph Edges:**")
        for u, v, data in B.edges(data=True):
            st.write(f"{u} - {v} : {data['weight']:.2f}")
    if show_metrics:
        st.markdown("**Graph Metrics:**")
        deg_centrality = nx.degree_centrality(B)
        st.write("Degree Centrality (sample):")
        for node, centrality in list(deg_centrality.items())[:5]:
            st.write(f"Node {node}: {centrality:.3f}")
        betw_centrality = nx.betweenness_centrality(B, weight='weight')
        st.write("Betweenness Centrality (sample):")
        for node, centrality in list(betw_centrality.items())[:5]:
            st.write(f"Node {node}: {centrality:.3f}")
    if show_popular:
        st.markdown("**Most Popular Books:**")
        degrees = B.degree()
        book_degrees = [(node, deg) for node, deg in degrees if B.nodes[node].get('bipartite') == 'books']
        top_books = sorted(book_degrees, key=lambda x: x[1], reverse=True)[:5]
        for book, deg in top_books:
            st.write(f"{book}: connected to {deg} users")
    if show_influencers:
        st.markdown("**Top Influential Users:**")
        betw_centrality = nx.betweenness_centrality(B, weight='weight')
        top_users = sorted(
            [(n, c) for n, c in betw_centrality.items() if B.nodes[n].get('bipartite') == 'users'],
            key=lambda x: x[1], reverse=True
        )[:3]
        for user, score in top_users:
            st.write(f"{user}: {score:.3f}")
    if plot_graph_flag:
        plot_bipartite_graph(B)
    
    # Compute similarity metrics.
    rating_matrix, user_similarity_df = compute_similarity(aggregated)
    
    st.header("Get Book Recommendations (Existing Users)")
    user_id = st.text_input("Enter your User ID", "U003")
    top_n = st.number_input("Number of recommendations", min_value=1, max_value=10, value=5, step=1)
    similar_users_count = st.number_input("Number of similar users to consider", min_value=1, max_value=10, value=3, step=1)
    if st.button("Get Recommendations"):
        recommendations = recommend_books(user_id, rating_matrix, user_similarity_df, top_n=top_n, similar_users_count=similar_users_count)
        if recommendations:
            st.subheader(f"Recommendations for {user_id}:")
            for book, score in recommendations:
                st.write(f"**{book}** — Score: {score:.2f}")
        else:
            st.error("User not found or no recommendations available.")
    
    st.header("Custom Profile Recommendations")
    st.write("Select the books you like to build your custom profile and get immediate recommendations.")
    custom_liked_books = st.multiselect("Select books you like", options=sorted(aggregated['book_id'].unique()))
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
            else:
                st.error("No recommendations available for your custom profile.")
    
    st.sidebar.header("All Users Recommendations")
    if st.sidebar.checkbox("Recommend for all users"):
        st.write("### Top Book Recommendations for All Users:")
        for user in rating_matrix.index:
            recs = recommend_books(user, rating_matrix, user_similarity_df, top_n=3)
            st.write(f"**{user}**:")
            for book, score in recs:
                st.write(f"- {book}: {score:.2f}")

if __name__ == "__main__":
    main()
