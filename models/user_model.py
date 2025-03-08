from mongoengine import Document, StringField

class User(Document):
    name = StringField(max_length=100) 
    phone = StringField(required=True, unique=True)
