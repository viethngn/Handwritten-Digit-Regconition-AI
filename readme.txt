Name: Viet H. Nguyen
ID: vienguyen
Class: CS210
Project#: 4
-------------------------------------------
The tree with criterion sum_absolute and median evaluation works best. However, the result is inconsistent since the partition of our test and train data is randomly selected.
The data in some field has large variance (ie. number or time pregnant, tricep thickness, serum insulin, diabetes pedigree) and thus may affect the accuracy of the result. One reasonable way to deal with this is to filter out any example that has a large variance in the data set before injecting it into the DT.
-------------------------------------------
I wrote a new class called Data_set_NN for NMIST data where it takes NMIST training_images and training_labels as training data and test_images and test_labels as test data. The input is a 28x28 images -> there are 784 features. The output is a 10-element array represent the label:
[1,0,0,0,0,0,0,0,0,0] = 0
[0,1,0,0,0,0,0,0,0,0] = 1
[0,0,1,0,0,0,0,0,0,0] = 2
...
[0,0,0,0,0,0,0,0,0,1] = 9
-------------------------------------------
Results are as follow:

*With 1 linear_complete_layer (10 outputs), 1 sigmoid_layer, 1 iteration:
Training set 60000 examples. Number of columns: {29} 
Test set 10000 examples. Number of columns: {28}
Accuracy: 0.098
Time: 427.6136456769891

*With 1 linear_complete_layer (10 outputs), 1 sigmoid_layer, 100 iterations: 
Currently running but is taking too long

*With 1 linear_complete_layer (392 outputs), 1 sigmoid_layer, 1 linear_complete_layer (10 outputs), 1 sigmoid_layer, 100 iterations: 

-------------------------------------------

