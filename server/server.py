import base64
import time
from multiprocessing import Pool

import redis
from tornado import websocket, web, ioloop

from recorder import create_videostream

MAX_FPS = 24

class IndexHandler(web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    """ Handler for the root static page. """
    def get(self):
        """ Retrieve the page content. """
        self.render('index.html')

class RestHandler(web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def get(self):
        self.write({'message': 'hello world'})

class SocketHandler(websocket.WebSocketHandler):
    """ Handler for the websocket URL. """

    def __init__(self, *args, **kwargs):
        """ Initialize the Redis store and framerate monitor. """

        super().__init__(*args, **kwargs)
        self._store = redis.Redis()
        self._prev_image_id = None

    def check_origin(self, origin):
        return True

    def on_message(self, message):
        """ Retrieve image ID from database until different from last ID,
        then retrieve image, de-serialize, encode and send to client. """

        while True:
            time.sleep(1./MAX_FPS)
            image_id = self._store.get('image_id')
            if image_id != self._prev_image_id:
                break
        self._prev_image_id = image_id
        image = self._store.get('image')
        image = base64.b64encode(image)
        self.write_message(image)

app = web.Application([
    (r'/', IndexHandler),
    (r'/ws', SocketHandler),
    (r'/test', RestHandler),
])

def create_app():
    app.listen(9000)
    ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    pool = Pool(processes=2)
    stream = pool.apply_async(create_app)
    server = pool.apply_async(create_videostream)

    pool.close()
    pool.join()