# A daisi library for augmenting imbalanced data

Quite often we are handed datasets that have imbalance. Majority of the data belongs to a specific category while only few belong to the other. Any classifier trained on this kind of data would nurture a bias towards the majority class. It would be great to artificially create data points similar to the minority data, to add a semblance of balance to the dataset. There are many libraries that achieve exactly that.

One of the popular ones is the smote-variant library which has around 85 algorithms that can be utilised to augment data. We expose this library through daisi.

## Supported functions


```
get_all_oversamplers()
```

This function returns a list of names of all oversampling algorithms present in the library (See bottom for the list).
 
You can choose any of the names and pass that to the below function to get the oversampler object.

```
get_oversampler_object_by_name(ovsampler_name)
```
Returns an oversampler object depending on the name of oversampler passed. The name has to be from the list obtained using the ```get_all_oversampler()``` function.

The object returned from the function can be used to perform the oversampling. Following function performs the same
```
oversample(X,y,ovsampler_object)
```

X and y are numpy matrices of numerical type.

Each of the oversamplers have parameters that govern the method of augmentation and the datapoints formed thereof. In order to change these parameters, there are two functions. The first one is 

```
get_oversampler_parameters(ovsampler_object)
```

which returns the parameters of the oversampler in the form of a dictionary. This can be used as a reference while changing the values.

The next fumnction is 
```
set_oversampler_parameter(ovsampler_object,param_dic)
```
which takes the oversampler object and a dictionary of parameters containing the altered values of parameters.

## Complete example

### Simple example with default parameters

```
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
X_new,y_new=over_samplers.over_samplers.oversample(X,y,os)
print("after oversampling, shape = ",X_new.shape,y_new.shape)

```

### Example with modified parameters


```
import pydaisi as pyd
import pandas as pd

over_samplers = pyd.Daisi("ashhadulislam/OverSampling")
df=pd.read_csv("iris_modified.csv")
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
# choose the oversampler	
name="MWMOTE"
os=over_samplers.get_oversampler_object_by_name(name)
dict_params=over_samplers.get_oversampler_parameters(os)
# change values of params
dict_params["proportion"]=2
os=over_samplers.set_oversampler_parameter(os,dict_params)
print("before oversampling, shape = ",X.shape,y.shape)
X_new,y_new=over_samplers.oversample(X,y,os)
print("after oversampling, shape = ",X_new.shape,y_new.shape)


```

List of all oversamplers:

Code to find all oversamplers supported in this library
```
import pydaisi as pyd
pyd.get_all_oversamplers()
```

Output:

```
['NoSMOTE', 'SMOTE', 'SMOTE_TomekLinks', 'SMOTE_ENN', 'Borderline_SMOTE1', 'Borderline_SMOTE2', 'ADASYN', 'AHC', 'LLE_SMOTE', 
'distance_SMOTE', 'SMMO', 'polynom_fit_SMOTE', 'Stefanowski', 'Safe_Level_SMOTE', 'MSMOTE', 'DE_oversampling', 'SMOBD', 'SUNDO', 'MSYN', 
'SVM_balance', 'TRIM_SMOTE', 'SMOTE_RSB', 'ProWSyn', 'SL_graph_SMOTE', 'NRSBoundary_SMOTE', 'LVQ_SMOTE', 'SOI_CJ', 'ROSE', 'SMOTE_OUT', 
'SMOTE_Cosine', 'Selected_SMOTE', 'LN_SMOTE', 'MWMOTE', 'PDFOS', 'IPADE_ID', 'RWO_sampling', 'NEATER', 'DEAGO', 'Gazzah', 'MCT', 'ADG', 
'SMOTE_IPF', 'KernelADASYN', 'MOT2LD', 'V_SYNTH', 'OUPS', 'SMOTE_D', 'SMOTE_PSO', 'CURE_SMOTE', 'SOMO', 'CE_SMOTE', 'ISOMAP_Hybrid', 
'Edge_Det_SMOTE', 'CBSO', 'DBSMOTE', 'ASMOBD', 'Assembled_SMOTE', 'SDSMOTE', 'DSMOTE', 'G_SMOTE', 'NT_SMOTE', 'Lee', 'SPY', 'SMOTE_PSOBAT', 
'MDO', 'Random_SMOTE', 'ISMOTE', 'VIS_RST', 'GASMOTE', 'A_SUWO', 'SMOTE_FRST_2T', 'AND_SMOTE', 'NRAS', 'AMSCO', 'SSO', 'DSRBF', 
'NDO_sampling', 'Gaussian_SMOTE', 'kmeans_SMOTE', 'Supervised_SMOTE', 'SN_SMOTE', 'CCR', 'ANS', 'cluster_SMOTE', 'E_SMOTE', 'ADOMS', 
'SYMPROD']
```


