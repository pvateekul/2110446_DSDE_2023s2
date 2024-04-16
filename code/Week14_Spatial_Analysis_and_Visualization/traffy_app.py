# taxi_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pydeck as pdk

# Function to load data
@st.cache_data
def load_data():
    df = pd.read_csv('taxi_od.csv', parse_dates=['time'])
    return df

df = load_data()

# Sidebar Filters
hour = st.sidebar.slider('Hour', min_value=0, max_value=23, value=(0, 23))
status = st.sidebar.radio('Status', ('All', 'O', 'D'))

# Filter data based on selections
filtered_df = df[(df['time'].dt.hour >= hour[0]) & (df['time'].dt.hour <= hour[1])]
if status != 'All':
    filtered_df = filtered_df[filtered_df['status'] == status]

# Slider to select the number of top clusters to display
num_top_clusters = st.sidebar.slider('N largest clusters to show', 1, 10, 5)

# Main app
st.title('Taxi OD Analysis')
st.write('### Number of pick-ups/drop-offs over time')

# Time series analysis of number of trips by hour
trips_by_hour = filtered_df.groupby(filtered_df['time'].dt.hour).size()
st.bar_chart(trips_by_hour)

# Time series analysis of average distance by hour
average_distance_by_hour = filtered_df.groupby(filtered_df['time'].dt.hour)['distance_meter'].mean()
st.write('### Average Distance (meters) by Hour')
st.bar_chart(average_distance_by_hour)

# Time series analysis of average duration by hour
average_duration_by_hour = filtered_df.groupby(filtered_df['time'].dt.hour)['duration_min'].mean()
st.write('### Average Duration (minutes) by Hour')
st.bar_chart(average_duration_by_hour)

# DBSCAN clustering
coords = filtered_df[['latitude', 'longitude']]
db = DBSCAN(eps=0.005, min_samples=10).fit(coords)
filtered_df['cluster'] = db.labels_

# Count the number of points in each cluster and identify the largest clusters
clusters_count = filtered_df['cluster'].value_counts()

# Exclude the '-1' cluster, which represents noise
clusters_count = clusters_count[clusters_count.index != -1]

# Get the cluster labels of the top clusters
top_clusters = clusters_count.iloc[:num_top_clusters].index.tolist()  

# Filter 'filtered_df' to keep only the points belonging to the top clusters
filtered_df = filtered_df[filtered_df['cluster'].isin(top_clusters)]

# Display top clusters
st.write('### Sizes of Clusters')
st.write(clusters_count.iloc[:num_top_clusters])

# Function to create basemap
def create_basemap():
    view_state = pdk.ViewState(
        latitude=13.7372,  
        longitude=100.5324,  
        zoom=10
    )
    basemap = pdk.Deck(
        layers=[],  # No layers
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/light-v9'
    )
    return basemap

# Function to create scatter map
def create_scatter_map(dataframe):

    # Generate a unique color for each cluster using a colormap
    unique_clusters = dataframe['cluster'].unique()
    num_clusters = len(unique_clusters)
    
    # Use a continuous colormap to generate colors, ensure we have enough colors for all clusters. Also set transparency value to 64 (25%).
    colormap = plt.get_cmap('hsv')
    cluster_colors = {cluster: [int(x*255) for x in colormap(i/num_clusters)[:3]] + [128] 
                      for i, cluster in enumerate(unique_clusters)}
    
    # Map cluster ID to color for each row in the dataframe
    dataframe['color'] = dataframe['cluster'].map(cluster_colors)


    # Define the scatter plot layer
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        dataframe,
        get_position="[longitude, latitude]",
        get_color='color',
        get_radius=200,
    )
    view_state = pdk.ViewState(
        latitude=dataframe['latitude'].mean(),
        longitude=dataframe['longitude'].mean(),
        zoom=10
    )
    scatter_map = pdk.Deck(layers=[scatter_layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/light-v9')
    return scatter_map

# Function to create heatmap
def create_heatmap(dataframe):
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        dataframe,
        get_position="[longitude, latitude]",
        #get_weight="distance_meter",
        radius_pixels=50,
    )
    view_state = pdk.ViewState(
        latitude=dataframe['latitude'].mean(),
        longitude=dataframe['longitude'].mean(),
        zoom=10
    )
    heatmap = pdk.Deck(layers=[heatmap_layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/light-v9')
    return heatmap

# Display the basemap
st.write('### Basemap')
basemap = create_basemap()
st.pydeck_chart(basemap)

# Display Scatter Maps
st.write('### Scatter Map')
scatter_map = create_scatter_map(filtered_df)
st.pydeck_chart(scatter_map)

# Display Heatmap for both Origins and Destinations combined
st.write('### Heatmap')
combined_heatmap = create_heatmap(filtered_df)
st.pydeck_chart(combined_heatmap)


