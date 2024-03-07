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

## Describing the Data

![Capture](https://github.com/pratyush1718/CSE151A-NBA-Dataset/assets/83377067/dcfc4bc6-bed8-477c-a431-0aaaf960a8ce)
![Capture2](https://github.com/pratyush1718/CSE151A-NBA-Dataset/assets/83377067/2c02c998-c1e9-49e2-878b-87e472724cd7)
![Capture3](https://github.com/pratyush1718/CSE151A-NBA-Dataset/assets/83377067/5766896b-b421-42ad-b59b-4b3e17c50e9e)
![Capture4](https://github.com/pratyush1718/CSE151A-NBA-Dataset/assets/83377067/3553a1d2-ce62-450a-be81-24714e8efce2)

## Model 1
As it's shown below, the loss for both train and validation steadily rose instead of decreasing. While it's possible that our model could make improvements in finding the optimal hyperparamaters, it's more likely that our model is not doing well because our data is heavily imbalanced. As shown in the confusion matrix, the model only predicts one class and this is most likely the class that is reflected in a majority of the dataset. If we choose to continue to with this model in the future, we could improve the model by tuning hyperparamters and balancing the data by resampling so the classes are more evenly present.

![image](https://github.com/pratyush1718/CSE151A-NBA-Dataset/assets/81598019/610633a2-0326-4112-9d26-1094457f647b)

## Next Two Models
Two more models for predicting NBA play outcomes could include a CNN and a DNN. A CNN and a DNN could be trained to learn spatial patterns and relationships within play descriptions or game context, leveraging features such as text embeddings. In order to accurately reflect and model the complexities of the data, there's a lot of things we could improve on to make a better neural network. One of the things we aim to improve is the major imbalance in the target classifications. No play especially has a lot of data, at around 200K, while other classes have data in the tens of thousands. In our next models, in order to solve the data imbalance and poor model performance, we're going to preprocess our data differently by merging Away Play and Home Play to just one column that holds all the Plays. This way, when we one hot encode, we will end up with 8 classifications instead of the 15 we have now. 

## Conclusion
Our analysis of the model trained on the NBA dataset reveals several key findings. The model architecture consists of multiple layers with sigmoid and softmax activations, and prior to training, the dataset undergoes preprocessing steps including feature selection, one-hot encoding, and feature normalization in order to analyse the "Play" column accordingly. However, during the training process, the model demonstrates an increase in loss over epochs, which indicates potential issues with vanishing or exploding gradients. The classification report and log loss highlight poor performance across various classes and a significant discrepancy between the training and testing errors, suggesting overfitting. This overfitting is likely due to the model's high complexity, as indicated by the large number of parameters. To improve  performance, we will need to resample the data and further refine the preprocessing steps.

