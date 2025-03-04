import socket
import time


UR5_IP = "10.149.230.168"
PORT = 30001


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((UR5_IP, PORT))


ur_script = """
def move_up():\n
    speedl([0.031128133, 0.0015817875, -0.00022845135, 0, 0, 0], a=0.5, t=1)
end
"""

s.sendall(ur_script.encode("utf-8"))
print("!")
time.sleep(1)  
s.close()
