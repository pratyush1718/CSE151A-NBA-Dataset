# CSE151A-NBA-Dataset

The data we will be using for the group project is from the dataset: https://www.kaggle.com/datasets/schmadam97/nba-playbyplay-data-20182019

Specifically we will be looking through play by play data from the 2018-2019 season. 

## About the Data & Our Goal:

The NBA play-by-play dataset for the 2018-2019 season offers records of every play that occurred during games throughout the season. Each entry in the dataset includes features like the game clock, shot clock, team names, play type, and play outcome. Now, a very obvious prediction task is predicting the winning team... right? Well, since the dataset provides information about the sequence of plays during games, predicting the winning team is rather trivial. This is because the winning team remains constant throughout the whole game (At least in this dataset, since the winningTeam column just gives the end result of the entire game). That is to say, if a game has 200 plays represented by 200 rows in our dataset for that game, trying to use ML to predict the winning team would not factor in the variety of the plays that are run. Since the winning team (which, again, is an end result), does not change between plays, we are not going to be preprocessing our data according to that task. Instead, we want to predict specific play outcomes, which we can reshape as a classification task using preprocessing. Our prediction task now becomes the following: "given a time during the game, the two teams that are playing (among other features), can we predict the play that a team will run?"

## Data exploration:
 
  We will plot the original data in the form of a histogram, pairplot, correlation matrix, and a heat map.
  We will also print out the list of features that our data starts out with prior to doing the preprocessing. 
  Note: due to the nature of play by play data, for many observations many of the features include null data.
  Note2: the data includes both categorical and numerical data.



## Data Preprocessing:

Identify columns that contain information irrelevant to the classification task, such as specific types of fouls, jump balls, etc. Remove these columns from the dataset using pandas' drop() function.

**Refactor Play Column:**

Analyze the "Play" column to identify distinct play outcomes. Create a mapping of each play to a unique category, such as "make," "miss," "rebound," "turnover," "foul," and "no-play." The significance of "no-play" here is a strategy we are going to implement to process frequent null data in the play column. We will use pandas apply() function to map each play outcome to its corresponding category, creating a new column for the refactored play vector. Then, we will convert the refactored play column into a one-hot encoded format, where each category becomes its own column. We also aim to one-hot encode other categorical data such as location, teams playing, etc. Lastly, we will also normalize and split the temporal data part of the dataset (such as game date, time of play, quarter number, secLeft, etc.)

By the end of our preprocessing step, we will have prepped the data to predict the play that a team will run at any given point in the game!


## Next Two Models
Two more models for predicting NBA play outcomes could include a CNN and a simple logistic regression. A CNN could be trained to learn spatial patterns and relationships within play descriptions or game context, leveraging features such as text embeddings. On the other hand, a simple logistic regression model could provide a baseline approach, especially if the features exhibit linear relationships with the target outcomes. While the CNN offers flexibility in capturing intricate patterns, the simple logistic regression provides interpretability and computational efficiency, making them both viable options for NBA play prediction.

## Conclusion
Our analysis of the model trained on the NBA dataset reveals several key findings. The model architecture consists of multiple layers with sigmoid and softmax activations, and prior to training, the dataset undergoes preprocessing steps including feature selection, one-hot encoding, and feature normalization in order to analyse the "Play" column accordingly. However, during the training process, the model demonstrates an increase in loss over epochs, which indicates potential issues with vanishing or exploding gradients. The classification report and log loss highlight poor performance across various classes and a significant discrepancy between the training and testing errors, suggesting overfitting. This overfitting is likely due to the model's high complexity, as indicated by the large number of parameters. To improve  performance, we will need to resample the data and further refine the preprocessing steps.

