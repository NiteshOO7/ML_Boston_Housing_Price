# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:40:36 2018

@author: nitesh.yadav
"""
import Housing_Price_Prediction as hpp
from sklearn.cross_validation import train_test_split
import visuals as vs

def main():
    # load data from csv file
    data, features, labels = hpp.DataLoad()
    # Explore dataset using Statistics and graphs
    hpp.ExploreData(data, features, labels)
    # split data into training and testing sets
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)
    # Produce learning curves for varying training set sizes and maximum depths
    vs.ModelLearning(features, labels)
    vs.ModelComplexity(features_train, labels_train)
    # Fit the training data to the model using grid search
    reg = hpp.fitModel(features_train, labels_train)
    print("Parameter 'max-depth' is {} for the optimal model".format(reg.get_params()['max_depth']))
    print("Enter number of houses for which you want to predict their prices:")
    num_of_client = int(input())
    client_data = []
    for i in range(num_of_client):
        print("Enter the fature values of client {}'s house in the sequence: number of rooms, lower class status and student teacher ratio:".format(i + 1))
        client_data.append(i)
        client_data[i] = list(map(float, input().rstrip().split()))
    for i, price in enumerate(reg.predict(client_data)):
        print("Predicted selling price for Client {}'s house: ${:,.2f}".format(i + 1, price))
    
if __name__ == "__main__":
    main()
    