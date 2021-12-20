import redis
from tornado import web, ioloop

from api.base import IndexHandler, RestHandler, SocketHandler

app = web.Application([
    (r'/', IndexHandler),
    (r'/ws', SocketHandler),
    (r'/test', RestHandler),
])

def create_app():
    app.listen(9000)
    ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    create_app()