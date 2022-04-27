# Neural Networks: Charity Analysis

## Overview

For this project, our goal was to use our knowledge of Machine Learning & Neural Networks to create an algorithm that can help predict if applicants will be funded by a charitable organization known as Alphabet Soup. During our implementation we did the following:

- Preprocessing the data (removing features, editing features, etc.)
- Encode using ```OneHotEncoder()```
- Splitting the dataset using ```train_test_split()```
- Declaring our input features, hidden layers and neurons in each layer
- Compile, fit, and evaluate our model 
- Optimize

## Results

When looking at optimizing our model it is easy to overlook features that we believe to be necessary, when in actuality all they are doing is confusing our model (noisy data can skew our prediction capabilities no matter how many hidden layers & neurons we add). In the original model we were able to achieve an accuracy of ~53% on training data & ~63% on testing data (underfitting). In this original model we achieved this by binning ```APPLICATION_TYPE``` and ```CLASSIFICATION```. Moving forward I decided to take a look at the statistics for ```ASK_AMT``` since in our original analysis it showed that it had 8747 unique values (significantly higher than all of our other features). Using ```application_df['ASK_AMT'].describe()``` showed that our mean = 2.77e+06, with a std = 8.71e+07. The way I approached this issue wasn't to simply get rid of the column all together, but to attempt to bin our values in a way that might help our model. To do this I used the following thought process:

```
bins = [1,10000,25000,100000,500000,1000000,5000000,10000000,50000000,np.inf]
labels=['1-10K','10K-25K','25K-100K','100K-500K','500K-1M','1M-5M','5M-10M','10M-50M','50M+']

application_df['ASK_AMT'] = pd.cut(application_df['ASK_AMT'],bins,labels=labels)
```

After making this adjustment I decided to look at how many applications were submitted by each organization and to potentially bin them as well to add back into the model. In all honesty, after playing with ```name_counts``` of each organization I decided to bin everything into the "Other" category if there were less than 100 applications submitted (there was no mathematical or significant reason for this. I simply made the choice because keeping too many resulted in the model running slower, although I'm sure there is an optimal number of organization names to keep). After preprocessing (or I guess...re-preprocessing) the data, it was time to adjust the structure of the neural network. After adjusting the model, ```number_input_features = len(X_train[0]) = 82 ```, which means we should tweak the amount of hidden layers & neurons we are using. Below you can see the adjustments made to the structure of the neural network along with our evaluation.

![Neural_Network_Structure](https://github.com/brand0j/Neural_Network_Charity_Analysis/blob/main/Resources/Neural_Network_Structure.PNG)

![Model_Evaluation](https://github.com/brand0j/Neural_Network_Charity_Analysis/blob/main/Resources/Model_Evaluation.PNG)

In best practice, it is common to use 2-3* the amount of neurons as we have features. There was an additional hidden layer added to the network to enhance the model's overal learning ability with a total of 240 neurons across the three layers. After trying out different activations for the hidden layers, ReLU seemed to return the best results. Epochs were kept low to reduce runtime of the model while further tweaks were being made (in theory I could have attained higher accuracy by letting this run for ~100 epochs). Overall the model reached an acccuracy of 75.41% which is a vast improvement on the original model while also achieving our target accuracy of 75%. Two things that could have improved the model's performance would be to include more features (NAME, ASK_AMT bins, etc.) or to increase the amount of epochs, but both of these become computationally expensive (doing a combination of these actually led to an error message in jupyter notebook saying that I didn't have the space required to perform the calculation). One thing to note is that our model isn't overfitting or underfitting since the performance between the training and testing datasets are roughly the same. 

## Summary 

The fact that our model isn't overfitting or underfitting leads me to believe that to make this model isn't the best one for our dataset, considering that to make it more accurate we would need a large amount of computation power, or additional features. At first I thought it would be worth trying a RandomForest(), but this yielded similar results (it should be noted that time was reduced significantly using a RandomForest()). If the data set contained int64 or float values for ```INCOME_AMT``` & ```ASK_AMOUNT``` the model could have picked up on the correlation between the two. For example, intuitively you would think that if the ratio ```ASK_AMT/INCOME_AMT >1``` that the application would be turned down since the organization is asking for more than what they generate. 
