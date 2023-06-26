#/bin/bash
#Author: Jaime Lorenzo Sanchez
#Version: 1.0

directory="AnalisisTitanic"
titanic_file="titanic.py"
# Create the directory AnalisisTitanic

if [ ! -d $directory ]; then
	mkdir $directory

fi

cd $directory

# Create file titanic.py
if [ ! -f $titanic_file ]; then
	touch $titanic_file
	echo '"""' >> $titanic_file
	echo 'Analysys of dataset titanic' >> $titanic_file
	echo "Author: Jaime Lorenzo Sanchez" >> $titanic_file
	echo "Version: 1.0" >> $titanic_file
	echo '"""' >> $titanic_file
	# Python packages
	echo "import pandas as pd" >> $titanic_file
	echo "import seaborn as sns" >> $titanic_file
	echo "import matplotlib.pyplot as plt" >> $titanic_file
	echo "from sklearn.model_selection import cross_val_score" >> $titanic_file
	echo "from sklearn.tree import DecisionTreeClassifier" >> $titanic_file
	echo "from sklearn.ensemble import RandomForestClassifier" >> $titanic_file
	echo "from sklearn.neighbors import KNeighborsClassifier" >> $titanic_file
	echo "from sklearn.svm import SVC" >> $titanic_file
	# Load dataset titanic
	echo "titanic = pd.read_csv('https://raw.githubusercontent.com/p72losaj/Datasets/main/titanic.csv')" >> $titanic_file
	# Titanic Preprocessing
	echo "titanic_clean = titanic.copy()" >> $titanic_file
	# Drop innecesary information
	echo "titanic_clean.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'],axis=1,inplace=True)" >> $titanic_file
	# Clean missing data
	echo "titanic_clean = titanic_clean.interpolate(method = 'linear', limit_direction = 'backward')" >> $titanic_file
	# Drop duplicate data
	echo "titanic_clean = titanic_clean.drop_duplicates(keep='last')" >> $titanic_file
	# Transform Sex Data
	echo "titanic_clean['Sex'] = titanic_clean['Sex'].replace(['female','male'],[0,1])" >> $titanic_file
	# Transform SibSp and Parch data
	echo "titanic_clean['inFamily'] = (titanic_clean['SibSp'] > 0) | (titanic_clean['Parch'] > 0)" >> $titanic_file
	echo "titanic_clean['inFamily'] = titanic_clean['inFamily'].astype(int)" >> $titanic_file
	echo "titanic_clean.drop(['SibSp', 'Parch'],axis=1,inplace=True)" >> $titanic_file
	# Transform age data
	echo "titanic_clean.loc[titanic_clean['Age']<=20, 'Age'] = 0" >> $titanic_file
	echo "titanic_clean.loc[(titanic_clean['Age'] >20) & (titanic_clean['Age'] <= 65), 'Age'] = 1" >> $titanic_file
	echo "titanic_clean.loc[ titanic_clean['Age'] > 65, 'Age'] = 2" >> $titanic_file
	echo "titanic_clean['Age'] = titanic_clean['Age'].astype(int)" >> $titanic_file
	# Transform fare data
	echo "titanic_clean.loc[ titanic_clean['Fare'] <= 7.925, 'Fare'] = 0" >> $titanic_file
	echo "titanic_clean.loc[ (titanic_clean['Fare'] > 7.925) & (titanic_clean['Fare'] <= 15.2458), 'Fare' ] = 1" >> $titanic_file
	echo "titanic_clean.loc[ titanic_clean['Fare'] > 15.2458,'Fare' ] = 2" >> $titanic_file
	echo "titanic_clean['Fare'] = titanic_clean['Fare'].astype(int)" >> $titanic_file
	# Transform survived data
	echo "titanic_clean['Survived'] = titanic_clean['Survived'].replace([1,0],['Alive','Dead'])" >> $titanic_file
	# Show result
	echo "print('Show titanic clean'); print(titanic_clean)" >> $titanic_file
	# Save titanic_clean
	echo 'titanic_clean.to_csv("titanic_clean.csv")' >> $titanic_file
	# Apply algorithms of mining data
	# Create models
	echo "models = [DecisionTreeClassifier(), KNeighborsClassifier(n_neighbors=3), RandomForestClassifier(), SVC(probability = True)]" >> $titanic_file
	echo "classifiers = []" >> $titanic_file
	echo "log_cols = ['Classifier', 'Accuracy']" >> $titanic_file
	echo "log = pd.DataFrame(columns=log_cols)" >> $titanic_file
	echo "acc_dict = {}" >> $titanic_file
        # train and test
	echo "i=0" >> $titanic_file
	echo "train_titanic = titanic_clean.drop(['Survived'], axis=1)" >> $titanic_file
	echo "train_titanic = pd.get_dummies(train_titanic)" >> $titanic_file
	echo "train_titanic = train_titanic.dropna(axis=1)" >> $titanic_file
	echo "test_titanic = titanic_clean['Survived']" >> $titanic_file
	echo "for model in models:" >> $titanic_file
	echo -e "\t name = model.__class__.__name__" >> $titanic_file
	echo -e "\t classifiers.append(name)" >> $titanic_file
	echo -e "\t acc = cross_val_score(model, train_titanic, test_titanic, cv=5, scoring='accuracy').mean()" >> $titanic_file
	echo -e "\t if name in acc_dict:" >> $titanic_file
	echo -e "\t \t acc_dict[name] += acc" >> $titanic_file
	echo -e "\t else:" >> $titanic_file
	echo -e "\t \t acc_dict[name] = acc" >> $titanic_file
	echo -e "\t log.loc[i, log_cols[0]] = models[i]" >> $titanic_file
	echo -e "\t log.loc[i, log_cols[1]] = acc_dict[name]" >> $titanic_file
	echo -e "\t i += 1" >> $titanic_file
	# Print result
	echo "print('Show results: ');print(log)" >> $titanic_file
	# Save result
	echo "plt.figure(figsize=(14,10))" >> $titanic_file
	echo "plt.xlabel('Accuracy')" >> $titanic_file
	echo "plt.title('Classifier Accuracy')" >> $titanic_file
	echo "sns.barplot(x=log.Accuracy, y=classifiers, data=log, color='b')" >> $titanic_file
        echo "plt.savefig('classifiers_titanic.png')" >> $titanic_file
	echo "plt.clf()" >> $titanic_file
fi

# Execute titanic.py
python3 $titanic_file
