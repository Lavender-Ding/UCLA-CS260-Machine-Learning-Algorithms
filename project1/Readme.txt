Read me

encoding.m        Function of one-hot encoding
knn_classify.m    Function of knn classifier
knn_test.m        Run and get the accuracy of training, validation and test data
boundarytest.m    Run and get the decision boundary for k=1,5,15,20

In order to get the accuracy of car_test, car_valid and car_test dataset classified by knn for k=1,3,5,…,23, run knn_test.m directly, the variable accuracy is the accuracy matrix. The first line shows k, the second line shows the train accuracy, the third line shows valid accuracy and the last line shows the corresponding test accuracy.

In order to get the decision boundary, run boundary test.m directly and you will get four figures showing the decision boundary for k=1,5,15,20.

Ps: In the inn-classify.m, when there are several labels that have been voted for the same time, I randomly choose a label to be the predicting label. Thus, the outcomes can be slightly different each time.

Name: Yi Ding
U-ID: 604588135