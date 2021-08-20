from django.shortcuts import render,redirect
import pandas as pd
#from .ml import model_final
# Create your views here.
#def home(request):
   # return render(request,'index.html')
import os
from django.conf import settings
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

file_path_table= os.path.join(settings.BASE_DIR,'Table_1.csv')
file_path_processed= os.path.join(settings.BASE_DIR,'processed table.csv')

from django.shortcuts import render, redirect
import os, random

# Finding attrition rate:
def index(request):
    name='the'
    if request.method == 'POST':
        if not request.POST.get('submit'):
            name = request.POST['person_name']
            location = request.POST['location']
            emp_group = request.POST['group']
            function = request.POST['func']
            gender = request.POST['gend']
            tenure_group= request.POST['ten']
            experience = request.POST['exp']
            age = request.POST['age']
            Maritial = request.POST['mar']
            Hiring = request.POST['hir']
            Promoted = request.POST['pro']
            Job = request.POST['mat']
            # print(name)

            results = Finder(name, location, emp_group, function, gender, tenure_group,
                                      experience, age, Maritial, Hiring, Promoted, Job)
            print(results)
            results = str(results[0])
            return render(request, 'results.html', {'result': results, 'name': name})
        else:
            print('Not Working')

    else:
        results = None

    return render(request, 'index.html')


def Finder(name, location, emp_group, function, gender, tenure_group,
                                  experience, age, Maritial, Hiring, Promoted, Job):
    if name != "":
        df = pd.DataFrame(columns=['id', 'Experience (YY.MM)', 'Age in YY.', 'New Location',
                                   'New Promotion', 'New Job Role Match', 'Agency', 'Direct',
                                   'Employee Referral', 'Marr.', 'Single', 'other status', 'B1', 'B2',
                                   'B3', 'other group', '< =1', '> 1 & < =3', 'Operation', 'Sales',
                                   'Support', 'Female', 'Male', 'other'])

        HiringSource = HiringPeep(Hiring)
        Maritial_Status = MStatus(Maritial)
        EmpGrp = EmployeeGrp(emp_group)
        tengrp = TenureGrp(tenure_group)
        func = FunctionName(function)
        gen = Gender(gender)
        count = Co()
        df2 = {'id': count, 'Experience (YY.MM)': float(experience), 'Age in YY.': float(age), 'New Location': location,
               'New Promotion': int(Promoted), 'New Job Role Match': int(Job), 'Agency': HiringSource[0],
               'Direct': HiringSource[1], 'Employee Referral': HiringSource[2], 'Marr.': Maritial_Status[0], 'Single':
               Maritial_Status[1], 'other status': Maritial_Status[2], 'B1': EmpGrp[0], 'B2': EmpGrp[1],
               'B3': EmpGrp[2], 'other group': EmpGrp[3], '< =1': tengrp[0], '> 1 & < =3': tengrp[1],
               'Operation': func[0], 'Sales': func[1], 'Support': func[2], 'Female': gen[0], 'Male': gen[1],
               'other': gen[2]}

        df = df.append(df2, ignore_index=True)

        # load the model from disk
        res = model.predict(df)
        print(res)

        return res

    else:
        return None

def Co():
    return random.randrange(20, 500)

def HiringPeep(x):
    if str(x) == "Agency":
        return [1, 0, 0] # Agency,Direct, Employee Referral
    elif str(x) == "Direct":
        return [0, 1, 0]
    else:
        return [0, 0, 1]


def MStatus(x):
    if str(x) == "Marr.":
        return [1, 0, 0]
    elif str(x) == "Single":
        return [0, 1, 0]
    else:
        return [0, 0, 1]

def EmployeeGrp(x):
    if str(x) == "B1":
        return [1, 0, 0, 0]
    elif str(x) == "B2":
        return [0, 1, 0, 0]
    elif str(x) == 'B3':
        return [0, 0, 1, 0]
    else:
        return [0, 0, 0, 1]

def TenureGrp(x):
    if str(x) == "< =1":
        return [1, 0]
    else:
        return [0, 1]

def FunctionName(x):
    if str(x) == "Operation":
        return [1, 0, 0]
    elif str(x) == "Sales":
        return [0, 1, 0]
    else:
        return [0, 0, 1]

def Gender(x):
    if str(x) == "Female":
        return [1, 0, 0]
    elif str(x) == "Male":
        return [0, 1, 0]
    else:
        return [0, 0, 1]


# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:26:38 2021

@author: Hamzah
"""
##Model
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.metrics import accuracy_score
import os

attrdata = pd.read_csv(file_path_table)
attrdata.drop(0, inplace=True)
attrdata.dropna(axis=0, inplace=True)
gender_dict = attrdata["Gender "].value_counts()

promoted_dict = attrdata["Promoted/Non Promoted"].value_counts()
func_dict = attrdata["Function"].value_counts()
Hiring_dict = attrdata["Hiring Source"].value_counts()
Marital_dict = attrdata["Marital Status"].value_counts()
Emp_dict = attrdata["Emp. Group"].value_counts()
Emp_dict['other group'] = 1
job_dict = attrdata["Job Role Match"].value_counts()
tenure_dict = attrdata["Tenure Grp."].value_counts()
location_dict = attrdata["Location"].value_counts()
location_dict_new = {
    'Chennai': 7,
    'Noida': 6,
    'Bangalore': 5,
    'Hyderabad': 4,
    'Pune': 3,
    'Madurai': 2,
    'Lucknow': 1,
    'other place': 0,
}


def location(x):
    if str(x) in location_dict_new.keys():
        return location_dict_new[str(x)]
    else:
        return location_dict_new['other place']


data_l = attrdata["Location"].apply(location)
attrdata['New Location'] = data_l

gen = pd.get_dummies(attrdata["Function"])

hr = pd.get_dummies(attrdata["Hiring Source"])


def Mar(x):
    if str(x) in Marital_dict.keys() and Marital_dict[str(x)] > 100:
        return str(x)
    else:
        return 'other status'


data_l = attrdata["Marital Status"].apply(Mar)
attrdata['New Marital'] = data_l

Mr = pd.get_dummies(attrdata["New Marital"])


def Promoted(x):
    if x == 'Promoted':
        return int(1)
    else:
        return int(0)


data_l = attrdata["Promoted/Non Promoted"].apply(Promoted)
attrdata['New Promotion'] = data_l

Emp_dict_new = {
    'B1': 4,
    'B2': 3,
    'B3': 2,
    'other group': 1,
}


def emp(x):
    if str(x) in Emp_dict_new.keys():
        return str(x)
    else:
        return 'other group'


data_l = attrdata["Emp. Group"].apply(emp)
attrdata['New EMP'] = data_l

emp = pd.get_dummies(attrdata["New EMP"])


def Job(x):
    if x == 'Yes':
        return int(1)
    else:
        return int(0)


data_l = attrdata["Job Role Match"].apply(Job)
attrdata['New Job Role Match'] = data_l


def Gen(x):
    if x in gender_dict.keys():
        return str(x)
    else:
        return 'other'


data_l = attrdata["Gender "].apply(Gen)
attrdata['New Gender'] = data_l
gend = pd.get_dummies(attrdata["New Gender"])
tengrp = pd.get_dummies(attrdata["Tenure Grp."])
dataset = pd.concat([attrdata, hr, Mr, emp, tengrp, gen, gend], axis=1)

dataset.drop(["table id", "name", "Marital Status", "Promoted/Non Promoted", "Function", "Emp. Group", "Job Role Match",
              "Location"
                 , "Hiring Source", "Gender ", 'Tenure', 'New Gender', 'New Marital', 'New EMP'], axis=1, inplace=True)

dataset1 = dataset.drop(['Tenure Grp.', 'phone number'], axis=1)

# dataset1.to_csv("processed table.csv")
dataset = pd.read_csv(file_path_processed)
dataset = pd.DataFrame(dataset)
y = dataset["Stay/Left"]
X = dataset.drop("Stay/Left", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

lr = LogisticRegression(C=0.1, random_state=42, solver='liblinear')
dt = DecisionTreeClassifier()
rm = RandomForestClassifier()
gnb = GaussianNB()
'''for a,b in zip([lr,dt,rm,supvm,knn,gnb],["Logistic Regression","Decision Tree","Random Forest","Naive Bayes"]):
    a.fit(X_train,y_train)
    prediction=a.predict(X_train)
    y_pred=a.predict(X_test)
    score1=accuracy_score(y_train,prediction)
    score=accuracy_score(y_test,y_pred)
    msg1="[%s] training data accuracy is : %f" % (b,score1)
    msg2="[%s] test data accuracy is : %f" % (b,score)
    print(msg1)
    print(msg2)'''

model = lr.fit(X_train, y_train)


