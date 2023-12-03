Lebedeva Daria d.lebedeva@innopolis.university Ds-01

## Movie Rating Prediction System
#### Overview
This report describes the development and implementation of a system designed to predict user ratings for movies. The system provides personalized predictions of how users might rate different movies, leveraging a collaborative filtering approach with the ALS (Alternating Least Squares) algorithm in Apache Spark's MLlib. The model is trained on a dataset containing user ratings for various movies, and it aims to recommend new movies to users based on their past ratings and preferences.

#### How to Use This Repository
To effectively utilize this repository, follow these steps:

##### Structure of the Repository
evaluate.py: Use this script to evaluate the trained model. It can also make predictions for a given user based on their favorite movies.
#### Running the Scripts
- Execute evaluate.py.
- Follow the prompts to input your favorite movies.
- The script will output movie recommendations based on your input.