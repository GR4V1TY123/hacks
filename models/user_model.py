from flask import Flask, render_template, request, jsonify
from flask_mongoengine import MongoEngine
from Server import db

class User(db.Document):
    name = db.StringField(required=True, max_length=100)
    phone = db.StringField(required=True, unique=True)