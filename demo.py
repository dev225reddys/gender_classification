from sklearn import tree
clf=tree.DecisionTreeClassifier()

#Challenge -create 3 more classifiers..
#1
#2
#3

#[height, weight, shoesize]
X = [[181,80,44],[177,70,43],[160,60,38],[154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']
#Challenge- ...and train them on our data
clf=clf.fit(X,Y)
prediction = clf.predict([[190,70,43]])
#Challenge compare their results and print the best one!
print(prediction)