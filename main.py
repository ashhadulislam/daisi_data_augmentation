import pandas as pd
import lib
import smote_variants as sv
import numpy as np



def get_all_oversamplers():
	'''
	gives a list of names of all oversamplers
	present in this library
	'''

	list_osamplers_name=[]
	list_oversamplers=sv.get_all_oversamplers()
	for l in list_oversamplers:		
		name=str(l).split("'>")[0].split(".")[-1]		
		list_osamplers_name.append(name)
	return list_osamplers_name


def get_oversampler_object_by_name(ovsampler_name):
	'''
	returns the oversampler object 
	according to the name of the oversampler passed
	'''

	if ovsampler_name not in lib.dic_oversamplers:
		print("Oversampler name incorrect/ not present among the following")
		print(list(lib.dic_oversamplers.keys()))
		return None
	return lib.dic_oversamplers[ovsampler_name]

def get_oversampler_parameters(ovsampler_object):
	'''
	returns a dictionary containing
	all the parameters specific to the oversampler passed 
	as parameter
	'''

	return ovsampler_object.get_params()

def set_oversampler_parameter(ovsampler_object,param_dic):
	'''
	used to modify the behaviour of an oversampler
	input arguments: the oversampler object and the dictionary of parameters
	central to that object
	if any of the keys in the dictionary are absent in the object
	no modification, returns the original oversampler object
	On success returns the oversaampler object with modified params
	'''

	for k,v in param_dic.items():
		
		if k not in ovsampler_object.get_params():
			print(k,"not among the parameters of ",str(ovsampler_object))
			print("returning oversampler object")
			return ovsampler_object
	if "random_state" in param_dic:
		if param_dic["random_state"]==None:
			param_dic["random_state"]=np.random
	ovsampler_object.set_params(**param_dic)
	return ovsampler_object
	



def oversample(X,y,ovsampler_object):
	'''
	performs oversampling by augmentation
	returns the oversampled X and y
	'''

	X_ov,y_ov=ovsampler_object.sample(X,y)
	return X_ov,y_ov




def test():
	'''
	test function
	not required
	'''
	
	df=pd.read_csv("iris_modified.csv")
	X=df.iloc[:,:-1].values
	y=df.iloc[:,-1].values
	print(X.shape,y.shape)

	# print(get_all_oversamplers())
	name="Borderline_SMOTE1"
	for name in list(lib.dic_oversamplers.keys()):
		print("*"*10)
		if name=="IPADE_ID":
			continue
		print(name)
		os=get_oversampler_object_by_name(name)
		print("1.os is",os)
		dict_params=get_oversampler_parameters(os)
		print(dict_params)
		X_new,y_new=oversample(X,y,os)	
		print("after oversampling",X_new.shape,y_new.shape)
		X=df.iloc[:,:-1].values
		y=df.iloc[:,-1].values
		dict_params["proportion"]=1.2
		os=set_oversampler_parameter(os,dict_params)
		print(get_oversampler_parameters(os))
		X_new,y_new=oversample(X,y,os)
		print("after oversampling with changed params",X_new.shape,y_new.shape)
		
		
		
		
		


if __name__=="__main__":
	df=pd.read_csv("iris_modified.csv")
	X=df.iloc[:,:-1].values
	y=df.iloc[:,-1].values
	# choose the oversampler	
	name="MWMOTE"
	os=get_oversampler_object_by_name(name)
	dict_params=get_oversampler_parameters(os)
	dict_params["proportion"]=2
	os=set_oversampler_parameter(os,dict_params)
	print("before oversampling, shape = ",X.shape,y.shape)
	X_new,y_new=oversample(X,y,os)
	print("after oversampling, shape = ",X_new.shape,y_new.shape)

