from flask import Flask
from nyancheck.controllers import check

app = Flask(__name__)
app.register_blueprint(check.app)

# 起動する
if __name__ == "__main__":
    app.run(host='0.0.0.0')
