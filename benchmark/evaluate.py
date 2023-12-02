from pyspark.sql import SparkSession, Row
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import col, lower,lit

def get_movie_id(movies_df, movie_title):
    """
    Helper function to get the movie ID from the movie title.
    Case-insensitive partial match. Returns None if the movie is not found.
    """
    try:
        # Debugging: Print the DataFrame schema to check column names
        movies_df.printSchema()

        # Filter the DataFrame for the movie title
        filtered_df = movies_df.filter(lower(col("movie_title")).contains(lower(lit(movie_title))))
        first_match = filtered_df.first()

        if first_match:
            return first_match['movie_id']
        else:
            print(f"No match found for '{movie_title}'")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    # Initialize Spark Session
    spark = SparkSession.builder.appName("MovieRecommendation").getOrCreate()

    # Load the trained model
    model = ALSModel.load('../models/1-model')

    # Load movies data
    movies_df = spark.read.csv('data/filtered_movies.tsv', header=True, inferSchema=True)

    # Ask user for their favorite movies
    user_favorites = input("Enter a list of your favorite movies separated by commas: ").split(',')

    # Map movie titles to IDs
    favorite_movie_ids = [get_movie_id(movies_df, title.strip()) for title in user_favorites]
    favorite_movie_ids = list(filter(None, favorite_movie_ids))  # Remove None values

    # If no valid movie IDs found
    if not favorite_movie_ids:
        print("No valid movie titles found.")
        return

    # Create a DataFrame for the user's favorite movies
    user_id = 0  # Using a dummy user ID
    user_movies_df = spark.createDataFrame([(user_id, movie_id) for movie_id in favorite_movie_ids], ["user_id", "movie_id"])

    # Generate recommendations
    recommendations = model.recommendForUserSubset(user_movies_df, 5)
    recommended_movie_ids = [row.movie_id for row in recommendations.collect()[0].recommendations]

    # Print recommended movie titles
    print("\nRecommended Movies:")
    for movie_id in recommended_movie_ids:
        movie_title = movies_df.filter(movies_df.movie_id == movie_id).first().movie_title
        print(movie_title)

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":

    main()