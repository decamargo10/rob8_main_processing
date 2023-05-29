import socket
import pickle
import cv2
import time
from PIL import Image
import cv2

class Server:
    def __init__(self):
        # Set up server
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("192.168.0.102", 12348))  # Bind to all IP addresses of this host
        s.listen(30)  # Listen for 1 connection

        # Accept a client connection
        self.conn, self.addr = s.accept()

    def get_robo_pose_image(self, vis=False):
        # Receive the size of the data
        data_size = int.from_bytes(self.conn.recv(4), byteorder='big')

        # Receive image data
        data = b""
        while len(data) < data_size:
            packet = self.conn.recv(4096)
            if not packet: break
            data += packet
        img = pickle.loads(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("recv_img.jpeg", img)
        # Display the image
        # Convert the numpy array image to a PIL image
        if vis:
            img_pil = Image.fromarray(img)
            # Display the image
            img_pil.show()

        return img

    def send_data(self, GazeX, GazeY, IntentionBool):
        #print("SERVER: sending ", GazeX, GazeY, IntentionBool)
        data = pickle.dumps((GazeX, GazeY, IntentionBool))
        try:
            self.conn.sendall(data)
        except:
            print("Could not send points to client.")

