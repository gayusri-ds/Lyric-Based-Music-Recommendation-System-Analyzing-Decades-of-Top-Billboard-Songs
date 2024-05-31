## Lyric-Based Music Recommendation System: Analyzing Decades of Top Billboard Songs

## Project Overview

This project aims to develop a music recommendation system based on the lyrical content of the top 100 songs from each year between 1959 and 2023. By leveraging natural language processing (NLP) techniques, the project will analyze song lyrics to extract meaningful features and build a recommendation engine that suggests songs with similar lyrical themes. This system will not only recommend songs from the same era but can also bridge the gap between different musical decades, providing users with a unique and immersive listening experience.

## About the Dataset

## Dataset

This dataset comprises the top 100 songs (including lyrics) for each year from 1959 to 2023 in the United States. The top 100 songs of each year were web-scraped from [Billboard Top 100](https://billboardtop100of.com), and the lyrics were fetched using the Genius API.

The file `all_songs_data_processed.csv` is the output from the starter notebook, which contains a cleaned version of the dataset with parts of speech data (e.g., verbs, nouns) already extracted from the lyrics. This file should serve as an easier starting point for those who want to dive directly into analysis.

The dataset used in this project is `all_songs_data_processed.csv`, sourced from Kaggle.The dataset can be accessed [https://www.kaggle.com/datasets/brianblakely/top-100-songs-and-lyrics-from-1959-to-2019]

## Data Description

The dataset has a size of (6500 x 12), where each row describes a song with the following 12 variables:

- **Album**: The album where the song is featured.
- **Album URL**: A link to the album.
- **Artist**: The main artist of the song.
- **Featured Artists**: Other artists featured in the song.
- **Lyrics**: The lyrics of the song.
- **Media Link**: A link to a media source (e.g., YouTube).
- **Rank**: The song's rank at the end of the year.
- **Release Date**: The release date of the song (year-month-day).
- **Song Title**: The title of the song.
- **Song URL**: A link to the song's lyrics on the Genius API.
- **Writers**: The writers of the song.
- **Year of Ranking**: The year the song was ranked.

## Preprocessing and Modeling

### Preprocessing Steps

1. **Loading Data**:
    - Load the dataset and examine its structure.
    - Handle any missing values appropriately.

2. **Text Cleaning**:
    - Remove unwanted characters and symbols from the lyrics.
    - Convert text to lowercase to ensure uniformity.

3. **Tokenization**:
    - Split the lyrics into individual tokens (words).

4. **Stop Words Removal**:
    - Remove common stop words that do not contribute significantly to the meaning.


### Modeling Steps

1. **Feature Extraction**:
    - Create features from the text data such as word counts, TF-IDF values, or word embeddings.

2. **Similarity Calculation**:
    - Calculate similarity between songs based on the extracted features using metrics such as cosine similarity.

3. **Recommendation System**:
    - Build a recommendation system to suggest songs based on their similarity to a given song.

## Tools and Technologies:

   - Python: For data processing, analysis, and modeling.
   - Pandas and NumPy: For data manipulation and analysis.
   - NLTK : For natural language processing tasks.
   - Scikit-learn: For building and evaluating the recommendation model.
   - Streamlit: For developing the web interface.
   - Google Colab: For exploratory data analysis and prototyping.