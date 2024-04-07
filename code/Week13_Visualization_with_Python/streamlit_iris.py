import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import plotly.express as px

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add sidebar widget for user input
st.sidebar.header('KMeans Clustering')
clusters = st.sidebar.slider('Select Number of Clusters:', 1, 10, 3)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=clusters, random_state=42)
iris_df['Cluster'] = kmeans.fit_predict(iris_df).astype(str)  

# Plotly Express scatter plot matrix
fig = px.scatter_matrix(iris_df,
                        dimensions=iris.feature_names,
                        color='Cluster',
                        title='Iris Dataset KMeans Clustering',
                        color_discrete_sequence=px.colors.qualitative.Set1) 

# Display the plot
fig.update_layout(height=800, width=800)
st.plotly_chart(fig)
