import socket
import pickle
import cv2


class Client:
    def __init__(self):
        # Connect to server
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(("192.168.0.102", 12348))

    def send_robot_image_path(self, path):
        # Load and send image
        img = cv2.imread(path)
        data = pickle.dumps(img)
        # Send the size of the data first
        self.s.sendall(len(data).to_bytes(4, byteorder='big'))
        self.s.sendall(data)

    def send_robot_image(self, img):
        data = pickle.dumps(img)
        # Send the size of the data first
        self.s.sendall(len(data).to_bytes(4, byteorder='big'))
        self.s.sendall(data)

    def recv_data(self):
        data = self.s.recv(1024)
        if data:
            x, y, boolean = pickle.loads(data)
            print(x, y, boolean)
            return x, y, boolean
        return None, None, None


client = Client()
img = cv2.imread("/home/decamargo/Documents/uni/yolov7/test_orig/images/valid_image_0000.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
client.send_robot_image(img)
while True:
    client.recv_data()
