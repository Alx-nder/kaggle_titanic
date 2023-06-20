import pandas as pd
import opendatasets as od

# od.download("https://www.kaggle.com/competitions/spaceship-titanic")

raw_data=pd.read_csv('spaceship-titanic/train.csv')
raw_data=raw_data[~raw_data['Cabin'].isnull()]

raw_data=raw_data[~raw_data['HomePlanet'].isnull()]
raw_data=raw_data[~raw_data['Destination'].isnull()]

extract_cabin=raw_data.Cabin.str.split('/',expand=True)
raw_data=raw_data.assign(Deck=extract_cabin[0])

ytrain=raw_data['Transported']



def clean_dt(raw_data):
    return (raw_data
    .drop(columns=['Name','PassengerId',])
    .assign(
        RoomService=raw_data.RoomService.fillna(0).replace('nan',0),
        CryoSleep=raw_data.CryoSleep.fillna(False),
        VIP=raw_data.VIP.fillna(False).replace('nan',0),
        FoodCourt=raw_data.FoodCourt.fillna(0).replace('nan',0),
        ShoppingMall=raw_data.ShoppingMall.fillna(0).replace('nan',0),
        Spa=raw_data.Spa.fillna(0).replace('nan',0),
        VRDeck=raw_data.VRDeck.fillna(0).replace('nan',0),
        Cabin=raw_data.Cabin.str.split('/',expand=True)[2]=='S',
        Age=raw_data.Age.fillna(raw_data.Age.median())
        )
     
    )

 # encoding boolean labels
from sklearn import preprocessing

def transform_step(raw_data):
    label_encoder=preprocessing.LabelEncoder()
    raw_data.CryoSleep=label_encoder.fit_transform(raw_data.CryoSleep).astype(int)
    raw_data.Cabin=label_encoder.fit_transform(raw_data.Cabin).astype(int)
    # raw_data.VIP=label_encoder.fit_transform(raw_data.VIP)
    raw_data=pd.get_dummies(raw_data,columns=['Destination','HomePlanet'])
    raw_data=pd.get_dummies(raw_data,columns=['Deck'])

    return raw_data

res_encoder=preprocessing.LabelEncoder()

ytrain=res_encoder.fit_transform(ytrain)

  
Xtrain=clean_dt(raw_data)
Xtrain=transform_step(Xtrain)
Xtrain.drop(columns=['Transported'],inplace=True)   

# cabins=Xtrain.Cabin.str.split('/',expand=True),
# cabins

raw_test=pd.read_csv('spaceship-titanic/test.csv')

extract_cabin=raw_test.Cabin.str.split('/',expand=True)
raw_test=raw_test.assign(Deck=extract_cabin[0])

Xtest=clean_dt(raw_test)
Xtest=transform_step(Xtest)

# Xtrain.drop(columns=['Deck_A','Deck_D','Deck_E','Deck_F','Deck_G','Deck_T'],inplace=True)
# Xtest.drop(columns=['Deck_A','Deck_D','Deck_E','Deck_F','Deck_G','Deck_T'],inplace=True)


# test train split
from sklearn import model_selection

kag_X_train,kag_X_test,kag_y_train,kag_y_test=model_selection.train_test_split(Xtrain,ytrain,test_size=.3,random_state=2,stratify=ytrain)
