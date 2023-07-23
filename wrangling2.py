import pandas as pd
import opendatasets as od

# od.download("https://www.kaggle.com/competitions/spaceship-titanic")

raw_data=pd.read_csv('spaceship-titanic/train.csv')

#remove raw null values 
# raw_data=raw_data[~raw_data['Cabin'].isnull()]
# raw_data=raw_data[~raw_data['HomePlanet'].isnull()]
# raw_data=raw_data[~raw_data['Destination'].isnull()]


ytrain=raw_data['Transported']


# function to handle other null values and do extra cleaning
def clean_dt(raw_data):
    return (raw_data
    .assign(
        RoomService=raw_data.RoomService.fillna(raw_data.RoomService.mode()[0]).replace('nan',raw_data.RoomService.mode()[0]),
        CryoSleep=raw_data.CryoSleep.fillna(raw_data.CryoSleep.mode()[0]),
        VIP=raw_data.VIP.fillna(raw_data.VIP.mode()[0]),
        FoodCourt=raw_data.FoodCourt.fillna(raw_data.FoodCourt.mode()[0]).replace('nan',raw_data.FoodCourt.mode()[0]),
        ShoppingMall=raw_data.ShoppingMall.fillna(raw_data.ShoppingMall.mode()[0]).replace('nan',raw_data.ShoppingMall.mode()[0]),
        Spa=raw_data.Spa.fillna(raw_data.Spa.mode()[0]).replace('nan',raw_data.Spa.mode()[0]),
        VRDeck=raw_data.VRDeck.fillna(raw_data.VRDeck.mode()).replace('nan',raw_data.VRDeck.mode()[0]),
        # column that contains info on side Port or Starboard as true(starboard) or false(Port)
        Starboard=raw_data.Cabin.str.split('/',expand=True)[2]=='S',
        Deck=raw_data.Cabin.str.split('/',expand=True)[0],

        # take median age
        Age=raw_data.Age.fillna(raw_data.Age.mode()[0])
        )     
    .drop(columns=['Name','PassengerId','Cabin'])

    )

 # encoding boolean labels to 0 or 1
from sklearn import preprocessing

def transform_step(raw_data):
    label_encoder=preprocessing.LabelEncoder()

    raw_data.CryoSleep=label_encoder.fit_transform(raw_data.CryoSleep).astype(int)
    raw_data.Starboard=label_encoder.fit_transform(raw_data.Starboard).astype(int)
    
    # raw_data.VIP=label_encoder.fit_transform(raw_data.VIP)
    
    raw_data=pd.get_dummies(raw_data,columns=['Destination','HomePlanet'])
    raw_data=pd.get_dummies(raw_data,columns=['Deck'])

    return raw_data


# encode validation
res_encoder=preprocessing.LabelEncoder()
ytrain=res_encoder.fit_transform(ytrain)

# clean train
Xtrain=clean_dt(raw_data)
# transform train
Xtrain=transform_step(Xtrain)
Xtrain.drop(columns=['Transported'],inplace=True)   


# test
raw_test=pd.read_csv('spaceship-titanic/test.csv')

# clean test
Xtest=clean_dt(raw_test)
# transform test
Xtest=transform_step(Xtest)

# test train split
from sklearn import model_selection

kag_X_train,kag_X_test,kag_y_train,kag_y_test=model_selection.train_test_split(Xtrain,ytrain,test_size=.25,random_state=2,stratify=ytrain)
