# streamlit run recommend_app.py
# cd 'MusicRecommender'


#pip install streamlit pandas numpy scikit-learn sentence-transformers

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess the dataset
df = pd.read_csv(r'Spotify-2000-lyrics-embedding.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
df['Lyrics_Embedding'] = df['Lyrics_Embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

# Load the BERT model
bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Setup song_id as index lookup
df['song_id'] = range(1, len(df) + 1)

# Setup df_normalized for Content-filtering recommender based on Audio Features
df_normalized = df[['song_id', 'BPM', 'Energy', 'Danceability',
       'Loudness', 'Liveness', 'Valence', 'Length', 'Acousticness',
       'Speechiness', 'Popularity']]

# Setup filtered_df for Lyrics-based Recommender
filtered_df = df[['song_id', 'Lyrics_Embedding']]

# Weights for combining the audio and lyrics similarity
audio_weight = 0.7
lyrics_weight = 1 - audio_weight



#PART 1: Run the Recommendation Model


def recommend_songs(song_id_or_query, df, df_normalized, filtered_df, num_songs=50):
    """
    Combines audio features and lyrics similarity to return song recommendations.
    """
    def content_filter_music_recommender(song_id, N):
        """Calculate audio similarity based on cosine similarity."""
        distance_method = cosine_similarity
        all_songs = df_normalized.index[df_normalized.index != song_id]
        distances = [
            distance_method(
                [df_normalized.loc[song_id].values],
                [df_normalized.loc[other_id].values]
            )[0][0]
            for other_id in all_songs
        ]
        # Create a DataFrame of results
        audio_similarity_df = pd.DataFrame({
            'song_id': all_songs,
            'audio_similarity': distances
        })
        return audio_similarity_df.sort_values(by="audio_similarity", ascending=False).head(N)

    # Determine if query is text or song_id
    if isinstance(song_id_or_query, int):
        # Use content-based filtering for song_id
        audio_recommendations = content_filter_music_recommender(song_id_or_query, num_songs)
        audio_recommendations = pd.merge(audio_recommendations, df, on="song_id", how="left")
    else:
        # Lyrics-based filtering for textual queries
        query_embedding = bert_model.encode(song_id_or_query)
        filtered_df['lyrics_similarity'] = filtered_df['Lyrics_Embedding'].apply(
            lambda x: cosine_similarity([query_embedding], [x])[0][0]
        )
        # Normalize lyrics similarity
        scaler = MinMaxScaler()
        filtered_df['lyrics_similarity'] = scaler.fit_transform(filtered_df[['lyrics_similarity']])
        # Merge with main DataFrame
        audio_recommendations = pd.merge(filtered_df, df, on="song_id", how="left")

    # Combine audio and lyrics similarities
    if 'audio_similarity' in audio_recommendations:
        audio_recommendations['Similarity'] = (
            audio_weight * audio_recommendations['audio_similarity'] +
            lyrics_weight * audio_recommendations.get('lyrics_similarity', 0)
        )
    else:
        audio_recommendations['Similarity'] = audio_recommendations['lyrics_similarity']

    # Sort recommendations and return top results
    recommendations = audio_recommendations.sort_values(by="Similarity", ascending=False).head(num_songs)
    return recommendations[['Title', 'Artist', 'Top Genre', 'Similarity', 'Lyrics']]  # Include 'Lyrics' column




# PART 2: Add Audio Analysis Dashboard
def audio_analysis_dashboard(recommendations, df):
    """
    Generate an analysis dashboard for the recommended songs.
    Visualizes features and explains recommendation logic.
    """
    st.markdown("## 🎧 Recommended Songs Analysis")
    st.markdown("Explore why these songs were recommended, based on their audio features.")
    
    # Select a song for detailed analysis
    selected_song = st.selectbox(
        "Select a song to view its audio feature breakdown:",
        recommendations['Title'].tolist()
    )

    # Find song details from the dataframe
    selected_song_data = df[df['Title'] == selected_song].iloc[0]

    # Audio features to analyze
    audio_features = ['BPM', 'Energy', 'Danceability', 'Loudness', 
                      'Liveness', 'Valence', 'Length', 'Acousticness', 
                      'Speechiness', 'Popularity']
    
    # Prepare data for visualization
    feature_values = [selected_song_data[feature] for feature in audio_features]
    feature_df = pd.DataFrame({
        "Feature": audio_features,
        "Value": feature_values
    })

    # Bar Chart for audio features
    st.markdown("### 🎵 Audio Features")
    st.bar_chart(feature_df.set_index("Feature")["Value"])

    # Explanation of recommendation logic
    st.markdown("### 📋 Recommendation Insights")
    st.write(
        f"The song **{selected_song}** was recommended because of its strong similarity in the following aspects:"
    )
    if 'Similarity' in recommendations.columns:
        # Highlight similar features based on recommendation logic
        similar_features = recommendations[recommendations['Title'] == selected_song].iloc[0]
        if 'audio_similarity' in similar_features:
            st.write("- **Audio Similarity**: High match in terms of energy, danceability, and other acoustic features.")
        if 'lyrics_similarity' in similar_features:
            st.write("- **Lyrics Similarity**: Matches the lyrical theme or sentiment of your query.")

    # Comparison with other recommended songs
    st.markdown("### 📊 Comparison with Other Songs")
    comparison_df = recommendations.merge(df, on="Title")[audio_features + ['Title']]
    st.dataframe(comparison_df.set_index('Title'))






# Generate a Spotify URL for embedding
def generate_spotify_url(title):
    song_query = f"{title}"
    spotify_search_url = f"https://open.spotify.com/search/{song_query.replace(' ', '%20')}"
    return spotify_search_url

# Streamlit app
def main():
    st.set_page_config(page_title="Music Recommendation", page_icon="🎵", layout="wide")
    
    # Spotify-like background and color
    st.markdown("""
        <style>
            body {
                background-color: #121212;
                color: #FFFFFF;
                font-family: 'Helvetica Neue', sans-serif;
            }
            .css-18e3th9 {
                background-color: #1DB954;
                color: white;
                border-radius: 20px;
                padding: 10px;
                font-size: 24px;
                text-align: center;
            }
            .css-1j8y4u8 {
                background-color: #121212;
            }
            .stButton button {
                background-color: #1DB954;
                color: white;
                border-radius: 20px;
                padding: 10px 20px;
                font-size: 18px;
                font-weight: bold;
                transition: 0.3s ease;
            }
            .stButton button:hover {
                background-color: #1aa34a;
            }
            .song-card {
                background-color: #333;
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
                transition: transform 0.3s ease;
            }
            .song-card:hover {
                transform: scale(1.05);
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
            }
            .song-title {
                font-size: 24px;
                font-weight: bold;
                color: #1DB954;
            }
            .song-genre {
                font-size: 16px;
                color: #B3B3B3;
            }
            .similarity-score {
                font-size: 18px;
                color: #B3B3B3;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("An AI-Music Assistant designed just for You 🎶")
    st.write("Let's enjoy personalized tracklists based on your query.")

    # Input fields
    num_songs = st.number_input("Numbers of songs you want in your list", min_value=1, max_value=50, value=5, step=1)
    query = st.text_input("Tell me what you want (e.g. 'Suggest me some chill songs/ I want some music like She will be loved/ I will be married soon!/ Im feeling happy!!!')")

    if st.button("Get Recommendations"):
        if query:
            with st.spinner("Finding recommendations..."):
                recommendations = recommend_songs(query, df, df_normalized, filtered_df, num_songs)
                if recommendations.empty:
                    st.write("No recommendations found. Try a different query.")
                else:
                    st.session_state['recommendations'] = recommendations
                    
                    st.write("### Here are your recommendations:")
                    for index, row in recommendations.iterrows():
                        song_title = row['Title']
                        spotify_url = generate_spotify_url(song_title)

                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"""
                                    <div class="song-card">
                                        <h3 class="song-title">{song_title}</h3>
                                        <p class="song-genre">Genre: {row['Top Genre']}</p>
                                        <p class="similarity-score">Similarity: {row['Similarity']:.3f}</p>
                                        <a href="{spotify_url}" target="_blank" style="color: #1DB954; font-weight: bold;">Play on Spotify</a>
                                    </div>
                                """, unsafe_allow_html=True)

                            with col2:
                                with st.expander("Show Lyrics"):
                                    st.write(row['Lyrics'])
        else:
            st.error("Please enter a query!")

    # Show the audio analysis dashboard
    if 'recommendations' in st.session_state:
        recommendations = st.session_state['recommendations']
        audio_analysis_dashboard(recommendations, df)
    else:
        st.write("Please get recommendations first to view the analysis.")

# Run the app
if __name__ == "__main__":
    main()