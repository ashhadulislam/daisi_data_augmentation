import pydaisi as pyd
import pandas as pd

over_samplers = pyd.Daisi("ashhadulislam/OverSampling")
df=pd.read_csv("iris_modified.csv")
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
# choose the oversampler	
name="Gaussian_SMOTE"
os=over_samplers.get_oversampler_object_by_name(name)
print("before oversampling, shape = ",X.shape,y.shape)
X_new,y_new=over_samplers.oversample(X,y,os).value
print("after oversampling, shape = ",X_new.shape,y_new.shape)