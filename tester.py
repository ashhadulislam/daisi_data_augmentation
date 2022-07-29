import pydaisi as pyd
import pandas as pd

over_samplers = pyd.Daisi("ashhadulislam/OverSampling")
df=pd.read_csv("iris_modified.csv")
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
# choose the oversampler	
name="MWMOTE"
os=over_samplers.get_oversampler_object_by_name(name)
res=over_samplers.get_oversampler_parameters(os)
dict_params=res.value
# change values of params
dict_params["proportion"]=2
os=over_samplers.set_oversampler_parameter(os,dict_params)
print("before oversampling, shape = ",X.shape,y.shape)
result=over_samplers.oversample(X,y,os)
X_new,y_new=result.value
print("after oversampling, shape = ",X_new.shape,y_new.shape)