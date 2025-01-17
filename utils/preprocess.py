import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from utils.data_utils import python_stratified_split

TRANSFORMER_EMBEDDING_DIM = 384

def get_goodbooks_10k():
    # Load the data
    books = pd.read_csv('goodbooks-10k/books.csv')
    ratings = pd.read_csv('goodbooks-10k/ratings.csv')

    # Filter all NaN values rows
    books = books.dropna()
    ratings = ratings.dropna()

    # Filter out the books not in the books list from rating
    ratings = ratings[ratings['book_id'].isin(books['book_id'])]

    # Filter out users with less than 100 ratings
    # user_counts = ratings['user_id'].value_counts()
    # user_ids = (user_counts[user_counts >= 100].index)
    # ratings = ratings[ratings['user_id'].isin(user_ids)]

    user_ids = ratings['user_id'].sample(n=1000, random_state=42).unique()
    ratings = ratings[ratings['user_id'].isin(user_ids)]
    
    # Filter out books with less than 10 ratings
    book_counts = ratings['book_id'].value_counts()
    book_ids = (book_counts[book_counts >= 30].index)
    books = books[books['book_id'].isin(book_ids)]
    ratings = ratings[ratings['book_id'].isin(book_ids)]

    # reindex the user ids
    user_id_map = {user_id: i for i, user_id in enumerate(ratings['user_id'].unique())}
    ratings['user_id'] = ratings['user_id'].map(user_id_map)

    # reindex the book ids
    book_id_map = {book_id: i for i, book_id in enumerate(books['book_id'])}
    ratings['book_id'] = ratings['book_id'].map(book_id_map)
    books['book_id'] = books['book_id'].map(book_id_map)

    assert(ratings['user_id'].nunique() == ratings['user_id'].max()+1)
    assert(ratings['book_id'].nunique() == ratings['book_id'].max()+1)
    assert(books['book_id'].nunique() == books['book_id'].max()+1)

    users = pd.DataFrame({'userid': range(len(user_ids))})

    # rename ratings user_id to userid
    ratings = ratings.rename(columns={'user_id': 'userid', 'book_id': 'itemid'})
    books = books.rename(columns={'book_id': 'itemid'})
    ratings['timestamp'] = 0

    # Load the pre-trained SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # randomize but repeat
    books_features_tensor = model.encode(books['original_title'].tolist())
    books_features_tensor = torch.tensor(books_features_tensor, dtype=torch.float)
    user_features_tensor = torch.randn(len(users), TRANSFORMER_EMBEDDING_DIM)

    col_user = 'userid'
    col_item = 'itemid'
    col_timestamp = 'timestamp'
    
    train_ratings, test_ratings = python_stratified_split(ratings, ratio=0.9, col_user=col_user, col_item=col_item, col_timestamp=col_timestamp)

    return users, books, train_ratings, test_ratings, user_features_tensor, books_features_tensor

def get_movielens_1m():
    user_columns = ['userid', 'gender', 'age', 'occupation', 'zipcode']
    movie_columns = ['movieid', 'title', 'genres']
    rating_columns = ['userid', 'movieid', 'rating', 'timestamp']
    users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, names=user_columns, engine='python', encoding='ISO-8859-1')
    movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, names=movie_columns, engine='python', encoding='ISO-8859-1')
    ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, names=rating_columns, engine='python', encoding='ISO-8859-1')

    # Filter the movies not in the ratings list
    movies = movies[movies['movieid'].isin(ratings['movieid'])]

    # Filter all NaN values
    users = users.dropna()
    movies = movies.dropna()

    # user reindexing
    user_to_index = {user: i for i, user in enumerate(users['userid'])}

    # reindex userid in users and ratings
    users['userid'] = users['userid'].map(user_to_index)
    ratings['userid'] = ratings['userid'].map(user_to_index)

    # movie reindexing
    movie_to_index = {movie: i for i, movie in enumerate(movies['movieid'])}

    # reindex movieid in movies and ratings
    movies['movieid'] = movies['movieid'].map(movie_to_index)
    ratings['movieid'] = ratings['movieid'].map(movie_to_index)

    assert(ratings['userid'].nunique() == ratings['userid'].max()+1)
    assert(ratings['movieid'].nunique() == ratings['movieid'].max()+1)
    assert(movies['movieid'].nunique() == movies['movieid'].max()+1)

    occupation_dict = {
        0: "other",
        1: "academic/educator",
        2: "artist",
        3: "clerical/admin",
        4: "college/grad student",
        5: "customer service",
        6: "doctor/health care",
        7: "executive/managerial",
        8: "farmer",
        9: "homemaker",
        10: "K-12 student",
        11: "lawyer",
        12: "programmer",
        13: "retired",
        14: "sales/marketing",
        15: "scientist",
        16: "self-employed",
        17: "technician/engineer",
        18: "tradesman/craftsman",
        19: "unemployed",
        20: "writer"
    }

    # Load the pre-trained SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for movie genres
    movies['genres list'] = movies['genres'].apply(lambda x: x.split('|'))
    movies['genres list'] = movies['genres list'].apply(lambda x: ' '.join(x))
    genres_embeddings = model.encode(movies['genres list'].tolist())

    # Generate embeddings for user occupations
    occupations_embeddings = model.encode(users['occupation'].apply(lambda x: occupation_dict[x]).tolist())

    # Convert embeddings to tensors
    movies_features_tensor = torch.tensor(genres_embeddings, dtype=torch.float)
    user_features_tensor = torch.tensor(occupations_embeddings, dtype=torch.float)

    ratings = ratings.rename(columns={'movieid': 'itemid'}) 
    movies = movies.rename(columns={'movieid': 'itemid'})

    col_user = 'userid'
    col_item = 'itemid'
    col_timestamp = 'timestamp'
    train_ratings, test_ratings = python_stratified_split(ratings, ratio=0.9, col_user=col_user, col_item=col_item, col_timestamp=col_timestamp)

    return users, movies, train_ratings, test_ratings, movies_features_tensor, user_features_tensor


def load_dataset(dataset):
    if dataset == 'goodbooks-10k':
        return get_goodbooks_10k()
    elif dataset == 'movielens-1m':
        return get_movielens_1m()
    else:
        raise ValueError('Dataset not found')