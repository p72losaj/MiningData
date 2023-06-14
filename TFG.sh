#/bin/bash
# Author: Jaime Lorenzo Sanchez
directory="MiningData"
loadFile="loadDataset.py"

if [ ! -d $directory ]; then
	mkdir $directory; echo "Directory $directory created"
	cd $directory
	if [ ! -f "segment_challenge.arff" ]; then
		wget https://raw.githubusercontent.com/p72losaj/Datasets/main/segment-challenge.arff
	fi
	if [ ! -f $loadFile ]; then
		touch $loadFile
		echo '"""' >> $loadFile
		echo "Fichero que obtiene los datos de distintos dataset" >> $loadFile
		echo "Autor: Jaime Lorenzo Sanchez" >> $loadFile
		echo "Version: 1.0" >> $loadFile
		echo '"""' >> $loadFile
		echo "# Python Packages" >> $loadFile
		echo "import pandas as pd" >> $loadFile
		echo "from scipy.io import arff" >> $loadFile
		echo "# titanic dataset" >> $loadFile
		echo "titanic = pd.read_csv('https://raw.githubusercontent.com/p72losaj/Datasets/main/titanic.csv')" >> $loadFile
		echo "print('Show titanic dataset \n', titanic.head())" >> $loadFile
		echo "# Iris dataset" >> $loadFile
		echo "iris = pd.read_csv('https://raw.githubusercontent.com/p72losaj/Datasets/main/iris.data',header=None, names=['sepal length','sepal width','petal length','petal width','Species'])" >> $loadFile
		echo "print('Show iris dataset \n', iris.head())" >> $loadFile
		echo "# Segment Challenge dataset" >> $loadFile
		echo "segment_challenge = pd.DataFrame(arff.loadarff('segment-challenge.arff')[0])" >> $loadFile
		echo "segment_challenge['class'] = segment_challenge['class'].str.decode('utf-8')" >> $loadFile
		echo "print('Show Segment Challenge dataset \n', segment_challenge.head())" >> $loadFile
		echo "File $loadFile created"
	fi

else
	cd $directory; python3 $loadFile
fi

