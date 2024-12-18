from flask import Flask
app = Flask(__name__)

# Import routes sau khi khởi tạo Flask app
from routes import *

if __name__ == '__main__':
    app.run(debug=True)


