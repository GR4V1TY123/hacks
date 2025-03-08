from mongoengine import Document, StringField

class User(Document):
    name = StringField(max_length=100, required=False)  # Optional name
    phone = StringField(unique=True, required=False)  # Optional phone number
    audio_url = StringField(required=False)  # Optional field to store Cloudinary audio URL
