import firebase_admin
from firebase_admin import credentials ,firestore


cred = credentials.Certificate("key/hackathon-11d72-firebase-adminsdk-7hnwq-45f6fa260b.json")
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'https://hackathon-11d72.firebaseio.com'
})

db = firestore.client()
for k in db.collection("test").get():
    print(k.id, k.to_dict())
