import pandas as pd
import ast
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Download resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# Initialize the WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Define stop words with nltk
stop_words = set(stopwords.words('english'))

# Load the training dataset
file_path = 'books_train.csv'
df = pd.read_csv(file_path)

# Function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(nltk_tag):
    tag_map = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
    return tag_map.get(nltk_tag[0], wordnet.NOUN) # Default to noun if no better match

# Function to preprocess text from 'description' column
def preprocess(text):    
    # 1. & 2. Remove extra blank spaces and convert to lowercase
    text = str(text).lower().strip()

    # Ensure a space after each period (if it isn't already there)
    text = re.sub(r'(?<=[^.])\.(?=\S)', '. ', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Split the text into individual words via tokenization
    words = nltk.word_tokenize(text)

    # 4. Remove stop words
    words = [word for word in words if word not in stop_words]

    # Get POS tags for each word
    pos_tags = nltk.pos_tag(words)

    # 5. Lemmatization
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    
    return words

# Function to extract genre from 'genres' column
def extract_genre(genre, index):
    if pd.isna(genre) or genre.strip() == "":
        return None
    try:
        genre_list = ast.literal_eval(genre)
        if isinstance(genre_list, list) and genre_list:
            if 0 <= index < len(genre_list):
                return genre_list[index]
    except (ValueError, SyntaxError):
        pass
    return None

# Keep only relevant columns (title, author, rating, description, language, genres, numRatings, coverImg)
columns = ['title', 'author', 'rating', 'description', 'language', 'genres', 'numRatings', 'coverImg']
df = df[columns]

# Drop rows with missing descriptions
df = df.dropna(subset=["description"])

# Apply genre function
df['firstGenre'] = df['genres'].apply(extract_genre, index=0)
df['secondGenre'] = df['genres'].apply(extract_genre, index=1)
df['thirdGenre'] = df['genres'].apply(extract_genre, index=2)

# Apply preprocess function
df['words'] = df['description'].apply(preprocess)

# Keep only English books
df = df[df['language'] == 'English']

# Save the result to a new CSV file
output_file = 'cleaned_books_train.csv'
df.to_csv(output_file, index=False)

print("Processing complete. Cleaned data saved to:", output_file)