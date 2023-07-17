#/bin/bash
#Author: Jaime Lorenzo Sanchez
#Version: 1.0

directory="AnalisisIris"
iris_file="iris.py"
# Create the directory AnalisisTitanic

if [ ! -d $directory ]; then
	mkdir $directory

fi

cd $directory

# Create file iris.py
if [ ! -f $iris_file ]; then
	touch $iris_file
	echo '"""' >> $iris_file
	echo 'Analysys of dataset titanic' >> $iris_file
	echo "Author: Jaime Lorenzo Sanchez" >> $iris_file
	echo "Version: 1.0" >> $iris_file
	echo '"""' >> $iris_file
	# Python packages
	echo "import pandas as pd" >> $iris_file
	echo "import seaborn as sns" >> $iris_file
	echo "import matplotlib.pyplot as plt" >> $iris_file
	echo "from sklearn.model_selection import cross_val_score" >> $iris_file
	echo "from sklearn.tree import DecisionTreeClassifier" >> $iris_file
	echo "from sklearn.ensemble import RandomForestClassifier" >> $iris_file
	echo "from sklearn.neighbors import KNeighborsClassifier" >> $iris_file
	echo "from sklearn.svm import SVC" >> $iris_file
	echo "from sklearn.metrics import confusion_matrix" >> $iris_file
	# Load dataset iris
	echo "# Load dataset iris" >> $iris_file
	echo "iris = pd.read_csv('https://raw.githubusercontent.com/p72losaj/Datasets/main/iris.data', header=None, names = ['sepal length','sepal width','petal length','petal width','Species'])" >> $iris_file
	# iris Preprocessing
	echo "# Iris preprocessing" >> $iris_file
	echo "iris_clean = iris.copy()" >> $iris_file
	# Clean missing data
	echo "iris_clean = iris_clean.interpolate(method = 'linear', limit_direction = 'backward')" >> $iris_file
	# Drop duplicate data
	echo "iris_clean = iris_clean.drop_duplicates(keep='last')" >> $iris_file
	# Show iris_clean
	echo "print('Show Iris clean'); print(iris_clean)" >> $iris_file
	# Save iris_clean
	echo 'iris_clean.to_csv("iris_clean.csv")' >> $iris_file
	# Apply algorithms of mining data
	echo "# Apply algorithms" >> $iris_file
	# Create models
	echo "models = [DecisionTreeClassifier(), KNeighborsClassifier(n_neighbors=3), RandomForestClassifier(), SVC(probability = True)]" >> $iris_file
	echo "classifiers = []" >> $iris_file
	echo "log_cols = ['Classifier', 'Accuracy']" >> $iris_file
	echo "log = pd.DataFrame(columns=log_cols)" >> $iris_file
	echo "acc_dict = {}; best_acc = 0; best_classifier=''" >> $iris_file
        # train and test
	echo "i=0" >> $iris_file
	echo "train_iris = iris_clean.drop(['Species'], axis=1)" >> $iris_file
	echo "test_iris = iris_clean['Species']" >> $iris_file
	echo "for model in models:" >> $iris_file
	echo -e "\t name = model.__class__.__name__" >> $iris_file
	echo -e "\t classifiers.append(name)" >> $iris_file
	echo -e "\t acc = cross_val_score(model, train_iris, test_iris, cv=5, scoring='accuracy').mean()" >> $iris_file
	echo -e "\t log.loc[i, log_cols[0]] = models[i]" >> $iris_file
	echo -e "\t log.loc[i, log_cols[1]] = acc" >> $iris_file
	echo -e "\t if acc > best_acc: " >> $iris_file
	echo -e "\t \t best_acc = acc; best_classifier = models[i]" >> $iris_file
	echo -e "\t i += 1" >> $iris_file
	# Print result
	echo "print('Show results: ');print(log)" >> $iris_file
	# Save result
	echo "plt.figure(figsize=(14,10))" >> $iris_file
	echo "plt.xlabel('Accuracy')" >> $iris_file
	echo "plt.title('Classifier Accuracy')" >> $iris_file
	echo "sns.barplot(x=log.Accuracy, y=classifiers, data=log, color='b')" >> $iris_file
        echo "plt.savefig('classifiers_iris.png')" >> $iris_file
	echo "plt.clf()" >> $iris_file
	# iris data analysis
	echo "# Analysis of data" >> $iris_file
	# Confusion Matrix
	echo "best_classifier.fit(train_iris,test_iris)" >> $iris_file
	echo "print('Show confusion matrix for best classifier: ', best_classifier)" >> $iris_file
	echo "print(test_iris.unique())" >> $iris_file
	echo "cm = confusion_matrix(test_iris, best_classifier.predict(train_iris))" >> $iris_file
	echo "print(cm)" >> $iris_file


fi

python3 $iris_file