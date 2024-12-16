# Music_Personalization_Recommender
Music Personalized Recommender that built based on Content-based Filtering and Lyric interpretion using NLP

The festive season is coming, and I want to build my own Music Personalized Recommendation that can most entertain my mood. You can also enjoy my Music Assistant as well.
Here is the link: https://daniel-music-personalized-recommender.streamlit.app/

**A. What's interesting about my Music Personalized Recommender?
1. Your Mood Interpreter:**
You can input what's your mood right now, like:
- I need some uplift music after a hard-working day
- I want some music like She will be loved
- It's Xmas time!
or...
- I will be married soon!

![image](https://github.com/user-attachments/assets/ca7a2c10-8cd7-42ba-ad99-2bccd8870c93)



2. Enjoy the song immediately:
- I have the "Play on Spotify" button so you can enjoy the song right away with just 1 click
- There is also a lyric by each song suggestions to help you sing right away!!
- Next update: I will try to help you play the song in my web-app. No need to go to Spotify!

![image](https://github.com/user-attachments/assets/1e768ff3-e4f1-473f-a1ec-cc6740c96847)


3. Analyse your mood and the songs matching
If you don't quite trust the recommender, or just need to know why you are matched with the songs? Here are the Analysis
- Analyse some Audio features: length, energy, BPM (tempo), danceability, etc. that seems to match your mood!
- Compare the songs you are analysing with other songs recommended to you - See how fit they are!

![image](https://github.com/user-attachments/assets/378cffb5-6291-4d83-828e-928be7de65bb)

4. Technical setup:
- To run the app, please download the requirements.txt, recommend_app.py & Spotify-2000-lyrics-embedding.csv
- For the analysis and recommendation models, please see the Spotify_Recommendation.ipynb

5. Algorithms behind (please see the Spotify_Recommendation.ipynb)
- From an open source csv file from Kaggle, I had the audio features with each Titles to perform Content-based Filtering
- I fetched the lyrics online from Lyrics Genius, and embedded the lyrics using BERT
- From that base, I built 2 recommender components: Audio-features and Lyrics then consolidate them into one final recommender using Weight. I tested several cases and feel like the recommender works best with Audio (70%) and Lyrics (30%) weight
- I also tried to evaluate my recommendation by using Clustering and PCA method. I tested some recommendation results with different queries. The results looked quite good!

![image](https://github.com/user-attachments/assets/90622aa0-e1d8-4c01-8843-f1a2c32f1b7f)

![image](https://github.com/user-attachments/assets/2cbd2eea-6830-473b-a6ba-7d626b383cad)

![image](https://github.com/user-attachments/assets/2e4478cc-4bb0-41d8-8a51-c866faedb52d)
