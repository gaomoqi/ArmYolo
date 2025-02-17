import socket
import time


UR5_IP = "10.149.230.168"
PORT = 30001


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((UR5_IP, PORT))


ur_script = """
def move_up():\n
    p = get_actual_tcp_pose()\n
    p[2] = p[2] - 0.05\n
    movel(p, a=1.2, v=0.05)\n
    textmsg(\"moveup\")\n

end
"""

s.sendall(ur_script.encode("utf-8"))
print("!")
time.sleep(1)  
s.close()
