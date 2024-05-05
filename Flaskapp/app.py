from flask import Flask, render_template
from trained import generate_output

app = Flask(__name__)

@app.route('/')
def index():
    output = generate_output()  
    return render_template('index.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)

