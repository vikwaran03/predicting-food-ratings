<title> Predicting food ratings with multiple features</title>

# Predicting food ratings with multiple features

EDA Website: [EDA_link](https://vikwaran03.github.io/eda_recipes/)

<h2> Framing the model </h2>

The question that my model will be trying to address is the following:<i><b> can we predict the rating of a food recipe using a classification model trained on features like:</b></i>
<ul>    
    <li> How long does a recipe take to make? </li>
    <li> Are there a lot of steps that make the recipe complicated? </li>
    <li> Do we need a lot of ingredients? </li>
    <li> How many calories is in a given recipe? </li>
    <li> What are the levels of macronutrients (fats, proteins, and carbohydrates)? </li>
</ul>

To solve this problem, I am going to be using a multiclass classification model that takes in feature data that correspond to the questions above and are used to make predictions on whether a certain recipe is a 1, 2, 3, 4, or 5 rating. The response variable in this case would be 'rating', which corresponds to the 'rating' column in the dataset. The reason I chose 'rating' as the response variable instead of a column like average_rating was because I wanted to use a classification model, since the value in 'rating' are ordinal categorical, it is very easy to see that a classifier would work great for this question. To evaluate this model, I am going to be using accuracy as the main metric. The reason I am using accuracy instead of F-1 score is that since this is not a binary classification problem, using a F-1 score would mean I would have to calculate the F-1 score for each different classification group, 1 through 5 in 'ratings', and then average all of those values. This would incoporate more bias and cause for error in evaluation rather than a simple metric like accuracy which just shows the proportion of correct classifications, regardless of the classification group. At the time of prediction, the model can only use the data that is readily available, which includes these main 5 features to make predictions: ('minutes', 'n_steps', 'n_ingredients', 'cals', 'fat', 'protein', 'carbs'). 


<h2> Baseline Model </h2>

Since the model is a classification problem, I found a good baseline model to be logistic regression. The first step in building this baseline model was feature engineering, which I did by standard scaling the 'fat', 'protein', and 'carbs' columns and also by binarizing the 'n_steps' and n_ingredients' columns. The main reason I binarized those two columns was because of my knowledge of eda from this site: [EDA_link](https://vikwaran03.github.io/eda_recipes/), which helped me realize that I could simplify this feature by setting the threshold at 9 and making all recipes with more than 9 steps/ingredients equal to 1, and everything else 0. The binarized data are all nominal and all other columns are quantitative. There wasn't much encoding I had to do for the baseline due to the lack of categorical variables present in the data. Now, after feature engineering, I built a sklearn pipeline that first did all of the feature engineering and preprocessing and then ran a logistic regression model on the data, with no specified hypeerparameters. The results of the model were pretty surprising since the little feature engineering I did led to pretty accurate results, with a training score of 0.77357 and a testing score of 0.77268. Hence, I would conclude that this logistic regression baseline model is a good model since it is pretty accurate on both seen and unseen data. It is a very good sign that the test score and training score are pretty similar to each other which shows that the model is not overfitting the data. 


<h2> Final Model </h2>

Now, time for feature engineering again! Since the baseline model was good, I decided to bring back the same features I created before into my final model. I created two new features, the first of which was for the 'cals' column. When I took a deeper dive into this column I saw that the distribution seemed to contain a lot of outliers and variance in the data, hence I thought if I can apply some function that would classify the data into 'low', 'mid', or 'hi' categories, I could then One-Hot-Encode the data to hopefully increase the model accuracy. So I created a pipeline that first used a FunctionTransformer that applied a function I created that returned a dataframe where the 'cals' column had values of 'hi' if the value was >= the 75th percentile, 'mid' if the value was >= 25th percentile and < 75th percentile, and 'low' if the value was <= the 25th percentile. After this, I One-Hot-Encoded that categorical data I created in the previous step. The other feature I created was the quantile transformer which I thought was a good idea after visualizing the distribution of the 'mins' column and thinking that I could eliminate the bias by trying to make the data more normal. These were the two new features I created, alongside my original features from the baseline model, that would be inputted into my model. 

The task of model selection was difficult, I tried a lot of different classification models including K-Neighbors Classification, Decision Tree Classifications, Random Forest Classifications, and even Support Vector Classifications. For each of the models above I ran GridSearchCV to find the best possible hyperparameters. I was able to get a successful result from K-Neighbors and Decision-Tree, but for Random Forest and Support Vector machines the search was taking way too long. Due to this, I went to comparing K-Neighbors and Decision-Tree and ultimately chose to use <b> K-Neighbors Classifications </b> since it was improving accuracy better than the Decision Tree, by the slightest margin, but also not overfitting the data too much. My final model is an improvement compared to the baseline model because it is making more correct predictions on both seen and unseen data. To quantify the improvement, I took the difference between the baseline accuracy and the final model accuracy to find out that in the dataset, the final model makes more than 220 correct predictions than the baseline on the training data and around 12 more on the testing data. Even though these numbers are not a lot improvement by any means, they are some validation that the model is an improvement compared to the baseline data. 


<h2> Fairness Analysis </h2>

To test the accuracy of my model in terms of fairness, I split my testing data into two groups. Group X is for when n_ingredients is less than 9 and Y is for when n_ingredients greater than or equal to 9. I will be evaluating these two groups based on the accuracy of the model when run on these two groups individually. Again, the motivation for these two groups was from the eda from the website linked in the Baseline Model section.

Null Hypothesis (H0): The classifier's accuracy is the same for both recipes with less than 9 ingredients and recipes with more than or equal to 9 ingredients, and any differences are due to chance.

Alternative Hypothesis (H1): The classifier's accuracy is higher for recipes with more than or equal to 9 ingredients.

The test statistic I will be using is the difference in accuracy (less minus more) and the significance level I will be evaluating on is alpha = 0.01.

After running the permutation tests, the resulting p-value is 0.002. Hence, the null hypothesis should be rejected and it can be said that there is a good chance the classifier's accuracy is higher for recipes with more than or equal to 9 ingredients. 

Here is a plot that shows the distribution of the calculated test statistics and the observed test statistic:


<iframe src="assets/plot1.html" width=800 height=600 frameBorder=0></iframe>
