Train Classifier
```
./interface2.sh C train ~/Desktop/IITD/COL761/COL761/A3/model-classification.pt ~/Desktop/IITD/COL761/COL761/A3/dataset/dataset_2/train ~/Desktop/IITD/COL761/COL761/A3/dataset/dataset_2/valid 
```

Evaluate Classifier
```
./interface2.sh C eval  ~/Desktop/IITD/COL761/COL761/A3/model-classification.pt ~/Desktop/IITD/COL761/COL761/A3/dataset/dataset_2/valid
```

Train Regressor
```
./interface2.sh R train ~/Desktop/IITD/COL761/COL761/A3/model-regression.pt ~/Desktop/IITD/COL761/COL761/A3/dataset/dataset_1/train ~/Desktop/IITD/COL761/COL761/A3/dataset/dataset_1/valid 
```


Evaluate Regressor
```
./interface2.sh R eval  ~/Desktop/IITD/COL761/COL761/A3/model-regression.pt ~/Desktop/IITD/COL761/COL761/A3/dataset/dataset_1/valid 
```