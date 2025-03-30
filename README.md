# Book-recommendation-system\documentclass[fontsize=11pt]{article}
\usepackage{amsmath}
\usepackage[utf8]{inputenc}
\usepackage[margin=0.75in]{geometry}

\title{CSC111 Project Proposal: BookNerd: A Graph-Based Recommendation System for Indie Literature}
\author{Abhinn Kaushik, Aadi Chauhan, Dennis Bince, Prabeer Singh Sohal}
\date{Tuesday, March 4, 2025}

\begin{document}
\maketitle

\section*{Problem Description and Research Question}

In today's digital world, readers are flooded with online reviews and ratings, which makes it hard to find great indie literature. Traditional recommendation systems, usually based on collaborative filtering, often miss the mark when it comes to capturing the unique tastes of readers who prefer independent and lesser-known books. With our project, \emph{BookNerd}, we want to tackle this problem by using a graph-based approach to model the relationships between readers and indie books. Our plan is to build a bipartite graph where one set of nodes represents users and the other set represents books, with edges showing interactions like reviews and ratings.

We will analyze this network using similarity measures such as the Jaccard index and common neighbor counts to identify groups of users with similar tastes and to discover hidden connections between books. By doing this, we hope to provide more personalized recommendations that traditional systems might miss.

\textbf{Project Question/Goal:} \textbf{Can a graph-based analysis of user-book interactions reveal distinctive patterns that lead to more personalized and accurate recommendations for indie literature enthusiasts?}

This problem matters because better recommendation systems can help emerging authors reach their audience and give readers a richer, more tailored book discovery experience. By studying the structure of these networks, our project aims to uncover how communities of readers form and how their interests connect, ultimately offering a new way to recommend indie books.

\section*{Computational Plan}

For our project, we will use a real-world dataset from a platform like Kaggle—specifically, an indie book reviews dataset. This dataset usually contains fields such as \texttt{user\_id}, \texttt{book\_id}, \texttt{rating}, and \texttt{review\_text}. For example, a typical record might look like: 
\begin{verbatim}
User: U1234, Book: B5678, Rating: 4.5, Review: "A refreshing narrative with deep, thought-provoking themes."
\end{verbatim}

Our computational plan for \emph{BookNerd} includes the following steps:

\begin{enumerate}
    \item \textbf{Data Preparation:}  
    \begin{itemize}
        \item Clean the dataset by removing noise, normalizing ratings, and handling any missing values.
        \item Preprocess the review text to extract sentiment and thematic insights.
    \end{itemize}
    
    \item \textbf{Graph Construction:}  
    \begin{itemize}
        \item Build a bipartite graph where users and books are represented as nodes, and the edges (weighted by rating scores) reflect user interactions.
        \item Optionally, create a secondary graph to capture similarities between books based on common user interactions.
    \end{itemize}
    
    \item \textbf{Graph Analysis and Computations:}  
    \begin{itemize}
        \item Use the \texttt{NetworkX} library to compute key graph metrics like degree centrality and betweenness centrality, and to perform community detection.
        \item Apply similarity algorithms (such as cosine similarity on rating vectors) to fine-tune the recommendation rankings.
        \item Aggregate the data to identify overall trends in user behavior and to see how clusters of readers and books form.
    \end{itemize}
    
    \item \textbf{Interactive Visualization:}  
    \begin{itemize}
        \item Utilize the \texttt{Plotly} library to develop an interactive network diagram.
        \item Use functions like \texttt{Scatter} for plotting nodes and \texttt{Figure} for customizing layouts, so that users can zoom, pan, and click on nodes to get more details.
    \end{itemize}
    
    \item \textbf{Algorithmic Enhancements and User Feedback:}  
    \begin{itemize}
        \item Implement iterative algorithms that update recommendations based on simulated user feedback, making the recommendation system dynamic.
    \end{itemize}
\end{enumerate}

By combining \texttt{NetworkX} for our graph computations with \texttt{Plotly} for interactive visualizations, our project not only meets the technical requirements but also offers an engaging way to explore the data and see how different books and readers connect.

\section*{References}

\begin{enumerate}
    \item Kaggle. \emph{Indie Book Reviews Dataset}. Retrieved from \texttt{https://www.kaggle.com/datasets/indie-book-reviews}.
    \item Plotly Technologies Inc. \emph{Plotly Python Open Source Graphing Library}. Retrieved from \texttt{https://plotly.com/python/}.
    \item NetworkX Developers. \emph{NetworkX Documentation}. Retrieved from \texttt{https://networkx.org/documentation/stable/}.
    \item Various scholarly articles on graph-based recommendation systems and community detection methodologies.
\end{enumerate}

\end{document}
