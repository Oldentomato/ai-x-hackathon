from tensorflow import keras
import firebase_admin
from firebase_admin import credentials ,firestore
import keyinfo as key

cred = credentials.Certificate("key/hackathon-11d72-firebase-adminsdk-7hnwq-45f6fa260b.json")
firebase_admin.initialize_app(cred,{
    'databaseURL' : key.keyURL
})

db = firestore.client()
for k in db.collection("test").get():
    print(k.id, k.to_dict())
    input = k.to_dict()


model = keras.models.load_model('/data/_1/mymodel')
result = model.predict(input)[0]

