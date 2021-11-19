from flask import Flask
from flask_restful import Resource, Api
import os

from api import graph
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'It':'works'}

api.add_resource(HelloWorld, '/')
api.add_resource(graph.Graph, '/graph')

if __name__ == '__main__':
    app.run(debug=True, port=os.environ['PORT'])