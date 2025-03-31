import pandas as pd
import networkx as nx
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from networkx.algorithms import community

def load_and_prepare_data(file_path):
    """
    Loads the CSV file, computes sentiment, normalizes ratings and sentiment,
    and aggregates interactions to produce a DataFrame for graph building.
    """
    df = pd.read_csv(file_path)
    
    # Calculate sentiment for each review using TextBlob
    df['sentiment'] = df['review'].apply(lambda review: TextBlob(review).sentiment.polarity)
    
    # Normalize the rating (assuming ratings are on a scale from 1 to 5)
    df['norm_rating'] = df['rating'] / 5.0
    
    # Normalize sentiment from [-1, 1] to [0, 1]
    df['norm_sentiment'] = (df['sentiment'] + 1) / 2
    
    # Calculate edge weight: 70% rating + 30% sentiment
    df['edge_weight'] = 0.7 * df['norm_rating'] + 0.3 * df['norm_sentiment']
    
    # If a user interacts with the same book more than once, average the edge weights.
    aggregated = df.groupby(['user_id', 'book_id'], as_index=False)['edge_weight'].mean()
    
    return aggregated

def build_bipartite_graph(aggregated):
    """
    Builds and returns a bipartite graph from the aggregated DataFrame.
    Users and books are nodes, with edges weighted by interaction strength.
    """
    B = nx.Graph()
    
    # Get unique user and book nodes
    user_nodes = aggregated['user_id'].unique()
    book_nodes = aggregated['book_id'].unique()
    
    # Add nodes with a bipartite attribute
    B.add_nodes_from(user_nodes, bipartite='users')
    B.add_nodes_from(book_nodes, bipartite='books')
    
    # Add edges between users and books with computed weight
    for _, row in aggregated.iterrows():
        B.add_edge(row['user_id'], row['book_id'], weight=row['edge_weight'])
    
    return B

def print_graph_edges(B):
    """
    Prints the edges in the bipartite graph along with their weights.
    """
    print("Edges in the bipartite graph with their assigned weights:")
    for u, v, data in B.edges(data=True):
        print(f"{u} - {v} : {data['weight']:.2f}")

def analyze_graph(B):
    """
    Calculates and prints key network metrics:
    degree centrality, betweenness centrality, and community structure.
    """
    # Degree centrality
    deg_centrality = nx.degree_centrality(B)
    print("\nDegree Centrality (sample):")
    for node, centrality in list(deg_centrality.items())[:5]:
        print(f"Node {node}: {centrality:.3f}")
    
    # Betweenness centrality (weighted)
    betw_centrality = nx.betweenness_centrality(B, weight='weight')
    print("\nBetweenness Centrality (sample):")
    for node, centrality in list(betw_centrality.items())[:5]:
        print(f"Node {node}: {centrality:.3f}")
    
    # Community Detection using Greedy Modularity
    communities = community.greedy_modularity_communities(B, weight='weight')
    print("\nDetected Communities:")
    for i, comm in enumerate(communities):
        print(f"Community {i+1}: {sorted(list(comm))}")

def compute_similarity(aggregated):
    """
    Builds a user-book rating matrix and computes the user-to-user cosine similarity matrix.
    Returns both the rating matrix and the similarity DataFrame.
    """
    # Build user-book rating matrix from aggregated data.
    rating_matrix = aggregated.pivot(index='user_id', columns='book_id', values='edge_weight').fillna(0)
    print("\nUser-Book Rating Matrix (first 5 rows):")
    print(rating_matrix.head())
    
    # Compute cosine similarity between users.
    user_similarity = cosine_similarity(rating_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)
    
    print("\nUser-to-User Cosine Similarity Matrix:")
    print(user_similarity_df.round(3))
    
    return rating_matrix, user_similarity_df

def recommend_books(user_id, rating_matrix, user_similarity_df, top_n=5, similar_users_count=3):
    """
    Generate personalized book recommendations for a given user based on similar users' interactions.
    
    Parameters:
      user_id (str): The target user's ID.
      rating_matrix (DataFrame): A user-book matrix with interaction weights.
      user_similarity_df (DataFrame): A user-to-user cosine similarity matrix.
      top_n (int): Number of top recommendations to return.
      similar_users_count (int): Number of similar users to consider.
      
    Returns:
      List of tuples: Each tuple contains (book_id, aggregated_score), sorted by score descending.
    """
    if user_id not in rating_matrix.index:
        print(f"User {user_id} not found in the rating matrix.")
        return []
    
    # Get similarity scores for the target user, excluding the user itself.
    sim_scores = user_similarity_df.loc[user_id].drop(user_id)
    top_similar_users = sim_scores.sort_values(ascending=False).head(similar_users_count)
    
    # Get books that the target user has already interacted with.
    target_books = set(rating_matrix.loc[user_id][rating_matrix.loc[user_id] > 0].index)
    
    # Aggregate weighted scores for candidate books.
    rec_scores = {}
    for similar_user, similarity in top_similar_users.items():
        for book, rating in rating_matrix.loc[similar_user].items():
            if rating > 0 and book not in target_books:
                rec_scores[book] = rec_scores.get(book, 0) + similarity * rating

    # Sort the candidate books by their aggregated weighted score.
    ranked_recs = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)
    
    return ranked_recs[:top_n]

if __name__ == "__main__":
    # Set the file path to your CSV file (update if necessary)
    file_path = 'sample_dataset.csv'
    
    # Step 1: Load data and prepare it.
    aggregated = load_and_prepare_data(file_path)
    
    # Step 2: Build the bipartite graph.
    B = build_bipartite_graph(aggregated)
    print_graph_edges(B)
    
    # Step 3: Analyze the graph.
    analyze_graph(B)
    
    # Step 4: Compute similarity metrics.
    rating_matrix, user_similarity_df = compute_similarity(aggregated)
    
    # Step 5: Test the recommendation function.
    test_user = "U003"  # Change this to any valid user ID in your dataset.
    recommendations = recommend_books(test_user, rating_matrix, user_similarity_df, top_n=5, similar_users_count=3)
    
    print(f"\nRecommendations for {test_user}:")
    for book, score in recommendations:
        print(f"{book}: {score:.2f}")
