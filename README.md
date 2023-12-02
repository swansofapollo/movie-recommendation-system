## Movie Rating Prediction System
#### Overview
This report describes the development and implementation of a system designed to predict user ratings for movies. The system provides personalized predictions of how users might rate different movies, leveraging a collaborative filtering approach with the ALS (Alternating Least Squares) algorithm in Apache Spark's MLlib. The model is trained on a dataset containing user ratings for various movies, and it aims to recommend new movies to users based on their past ratings and preferences.

#### How to Use This Repository
To effectively utilize this repository, follow these steps:

##### Prerequisites
- Apache Spark: The project is developed using Apache Spark. Ensure you have Spark installed and properly set up in your environment.
- Python: A working Python environment is necessary, with dependencies installed as per the requirements.txt file in this repository.
##### Structure of the Repository
evaluate.py: Use this script to evaluate the trained model. It can also make predictions for a given user based on their favorite movies.
#### Running the Scripts
- Execute evaluate.py with the path to the trained model and the movie dataset.
- Follow the prompts to input your favorite movies.
- The script will output movie recommendations based on your input.