# CSE151A-NBA-Dataset

The data we will be using for the group project is from the dataset: https://www.kaggle.com/datasets/schmadam97/nba-playbyplay-data-20182019

Specifically we will be looking through play by play data from the 2018-2019 season. We could not upload the dataset to Github since it was too large so we've provided the link instead. 


## Introduction:

Nationally as well as globally, basketball is one of the most popular sports to watch for entertainment. Like with many other forms of entertainment, fans and enthusiasts alike find predicting game outcomes a fun activity, like the March Madness brackets that people fill out in preparation for the NCAA basketball tournaments. And it's no wonder people want to predict the outcome of games; there's many different factors and strategies that combine together to make the outcome of a game challenging to predict. For this reason, we were interested in creating a model to predict the outcome of a play-by-play basketball dataset because we believe that it would challenge us and allow us to apply the skills we learned practically. Additionally, sports analytics has the potential to have a direct impact in increasing a team's likelihood of winning so it's a useful area to apply analytics. A good model would not only be accurate in training, but also have a good accuracy on unseen data because this means it would generalize well to real world data. 

The NBA play-by-play dataset for the 2018-2019 season offers records of every play that occurred during games throughout the season. Each entry in the dataset includes features like the game clock, shot clock, team names, play type, and play outcome. Now, a very obvious prediction task is predicting the winning team... right? Well, since the dataset provides information about the sequence of plays during games, predicting the winning team is rather trivial. This is because the winning team remains constant throughout the whole game (At least in this dataset, since the winningTeam column just gives the end result of the entire game). That is to say, if a game has 200 plays represented by 200 rows in our dataset for that game, trying to use ML to predict the winning team would not factor in the variety of the plays that are run. Since the winning team (which, again, is an end result), does not change between plays, we are not going to be preprocessing our data according to that task. Instead, we want to predict specific play outcomes, which we can reshape as a classification task using preprocessing. Our prediction task now becomes the following: "Given a time during the game, the two teams that are playing (among other features), can we predict the play that a team will run?"

<p align="center">
Figure 1: Our Dataset Example
 <img width="1579" alt="Screenshot 2024-03-09 at 9 56 32 AM" src="https://github.com/pratyush1718/CSE151A-NBA-Dataset/assets/81598019/f258dfb6-a04b-4317-9cc8-b1ebd43b5c21">
</p>
</br>
In a paper entitled "Modeling basketball play-by-play data" by Vračar, Štrumbelj, and Kononenko, the authors provide a methodology in producing a realistic simulation of a basketball game between two teams. Their focus was on capturing the context of the game at any given moment and then generalizing that to the broader game context of which team wins and which team loses. By fine tuning a generalized model, they were later able to successfully apply it to three seasons of NBA games with better predictive accuracy than other existing models. While this paper could be helpful in improving our model, there's some differences. Our data features a whole season so it has multiple teams, unlike Vračar, Štrumbelj, and Kononenko's model which only focuses on two teams at a time. Their general model was also not built on any real world data unlike our model, which will be dependent on the 2018-2019 NBA season data. And lastly, as shown in the figure below, their model has quite a significance on time where intervals between plays are highly relevant. While our model uses time, it's not emphasized as it's not the only feature we use.

<p align="center">
Figure 2: Generated Graphs from Two Models that Emphasize Time
</br>
<img width="614" alt="Screenshot 2024-03-09 at 10 31 03 AM" src="https://github.com/pratyush1718/CSE151A-NBA-Dataset/assets/81598019/641b5b7a-dd94-46e2-b749-9f1ccdd59dab">
</p>

Paper citation:
Petar Vračar, Erik Štrumbelj, Igor Kononenko, Modeling basketball play-by-play data, Expert Systems with Applications, Volume 44, 2016, Pages 58-66, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2015.09.004.



## Methods

### Data Exploration:

Below are histograms and a pairplot that describe the data. 
We found that due to the nature of play-by-play data, many observations in several columns include null data.

#### Describing the Data

<p align="center">
Figure 3: Histogram of Numerical Columns in Our Dataset
</br>
<img width="600" alt="Screenshot 2024-03-10 at 7 27 53 PM" src="https://github.com/pratyush1718/CSE151A-NBA-Dataset/assets/81598019/2d2a02e4-f98e-4b36-902a-b085bd3c8325">
</p>

<p align="center">
Figure 4: Histogram of Counts for Numerical Columns
</br>
<img width="600" alt="Screenshot 2024-03-10 at 7 31 19 PM" src="https://github.com/pratyush1718/CSE151A-NBA-Dataset/assets/81598019/b31b0a08-ea33-465f-bdb3-1ca2ffe6c420">
</p>

<p align="center">
Figure 5: Pairplots for Columns Quarter, SecLeft, AwayScore, and HomeScore
</br>
<img width="600" alt="Screenshot 2024-03-10 at 7 31 40 PM" src="https://github.com/pratyush1718/CSE151A-NBA-Dataset/assets/81598019/1da90025-c406-4a11-92ea-28aba056125e">
<img width="600" alt="Screenshot 2024-03-10 at 7 31 59 PM" src="https://github.com/pratyush1718/CSE151A-NBA-Dataset/assets/81598019/20c5ffc4-be89-43a2-9856-c79ef301ea29">
</p>

### Data Preprocessing:

Identify columns that contain information irrelevant to the classification task, such as specific types of fouls, jump balls, etc. Remove these columns from the dataset using pandas' drop() function.

**Refactor Play Column:**

Analyze the "Play" column to identify distinct play outcomes. Create a mapping of each play to a unique category, such as "make," "miss," "rebound," "turnover," "foul," and "no-play." The significance of "no-play" here is a strategy we are going to implement to process frequent null data in the play column. We will use pandas apply() function to map each play outcome to its corresponding category, creating a new column for the refactored play vector. Then, we will convert the refactored play column into a one-hot encoded format, where each category becomes its own column. We also aim to one-hot encode other categorical data such as location, teams playing, etc. Lastly, we will also normalize and split the temporal data part of the dataset (such as game date, time of play, quarter number, secLeft, etc.)

By the end of our preprocessing step, we will have prepped the data to predict the play that a team will run at any given point in the game!


### Model One
A DNN was implemented using Keras. The DNN features an input layer, two hidden layers, and one output layer. The two hidden layers both use the sigmoid activation function while the output layer uses softmax. The chosen optimizer was Adam, loss was categorical cross entropy, and the model was fit on the training data with a validation split of 0.2. A summary of the model is shown below.

<p align="left">
<img width="600" alt="Screenshot 2024-03-10 at 8 05 55 PM" src="https://github.com/pratyush1718/CSE151A-NBA-Dataset/assets/81598019/4b208fbc-82af-4301-9979-a3b1be3e9cc1">
</p>

### Model Two
Model Two was designed as a DNN using the Keras framework. It consists of an input layer, eight hidden layers, and an output layer. The hidden layers are structured with varying numbers of neurons, including 16, 32, 64, 128, 256, 128, 64, and 32 respectively. Each hidden layer, with the exception of the final one, uses the ReLU activation function to introduce non-linearity into the model. The output layer employs the softmax activation function to produce probabilities for each class. The Adam optimizer was employed, along with the categorical cross-entropy loss function. During training, the dataset was split into training and validation sets, with a 20% validation split. Additionally, early stopping was utilized to prevent overfitting, using 10 epochs. 
Summary of the model:
![Capture](https://github.com/pratyush1718/CSE151A-NBA-Dataset/assets/83377067/b971cbcd-9ad1-4e1e-a0fb-83606672f791)


### Model Three
"



## Results

### Model One
As it's shown below, the loss for both train and validation steadily rose instead of decreasing. While it's possible that our model could make improvements in finding the optimal hyperparamaters, it's more likely that our model is not doing well because our data is heavily imbalanced. As shown in the confusion matrix, the model only predicts one class and this is most likely the class that is reflected in a majority of the dataset. This aligns with the log losses for train (~4.45) and test (~24.03) with the loss for train being much lower than for test. If we choose to continue to with this model in the future, we could improve the model by tuning hyperparamters and balancing the data by resampling so the classes are more evenly present.

![image](https://github.com/pratyush1718/CSE151A-NBA-Dataset/assets/81598019/610633a2-0326-4112-9d26-1094457f647b)

#### Next Two Models
Two more models for predicting NBA play outcomes could include a CNN and a DNN. A CNN and a DNN could be trained to learn spatial patterns and relationships within play descriptions or game context, leveraging features such as text embeddings. In order to accurately reflect and model the complexities of the data, there's a lot of things we could improve on to make a better neural network. One of the things we aim to improve is the major imbalance in the target classifications. No play especially has a lot of data, at around 200K, while other classes have data in the tens of thousands. In our next models, in order to solve the data imbalance and poor model performance, we're going to preprocess our data differently by merging Away Play and Home Play to just one column that holds all the Plays. This way, when we one hot encode, we will end up with 8 classifications instead of the 15 we have now. 

### Model Two
Looking at the losses from our previous model and especially the graph with respect to training epochs, we see an unintuitive visual. The loss, instead of decreasing over time, actually increases. We see numerous causes for concern and improvements. One improvement we wish to make with this milestone is to modify our data processing. As seen in our previous notebook, there was a huge data imbalance which was further exaggerated by having 15 classes for our model to predict. In this milestone, we address both the issues by changing the output dimension of our model to (n, 1) instead of (2n, 2). More specifically, we are merging the away team play and home team play into one column and decreasing the occurrences of “no play.” Moreover, we will also be trying a DNN instead of a regular neural network to better capture the complexities for this problem.

![Capture](https://github.com/pratyush1718/CSE151A-NBA-Dataset/assets/83377067/75481dc5-d57a-4a10-84fb-9e8d5a3667b2)

### Model Three




## Discussion

### Model One
Our analysis of the model trained on the NBA dataset reveals several key findings. The model architecture consists of multiple layers with sigmoid and softmax activations, and prior to training, the dataset undergoes preprocessing steps including feature selection, one-hot encoding, and feature normalization in order to analyse the "Play" column accordingly. However, during the training process, the model demonstrates an increase in loss over epochs, which indicates potential issues with vanishing or exploding gradients. The classification report and log loss highlight poor performance across various classes and a significant discrepancy between the training and testing errors, suggesting overfitting. This overfitting is likely due to the model's high complexity, as indicated by the large number of parameters. To improve  performance, we will need to resample the data and further refine the preprocessing steps.

### Model Two
Our second model is a neural network classifier designed to predict various play types in basketball games based on our dataset containing features such as game type, location, quarter, time remaining, scores, and play descriptions. The model achieved decent accuracy after training for 15 epochs, with approximately 26.56% on the training set and 26.37% on the validation set. The classification report reveals some varience in the model's performance across different play types, with relatively higher precision and recall for predicting "play_make" and "play_turnover," but lower scores for classes like "play_foul" and "play_rebound." To improve the model we can include other features to incorporate additional relevant features such as time remaining in the quarter and player statistics. Also, we can experiment with different neural network architectures and hyperparameters to help find a more suitable model configuration. Our model 2 shows improvement over our model 1 by achieving stable losses and predicting multiple classes, with modest accuracy. While both models have data imbalance, model 2 attempts to address this issue through further one-hot encoding, showing our attempt to handle our data sets challenges that we saw in model 1.

### Model Three





## Conclusion





## Collaboration
What each group member contributed to this project.

#### Kesar Arora


#### Pratyush Chand


#### Jaeda Gantulga


#### Hannah Ghassemian


#### Pranav Mekkoth





