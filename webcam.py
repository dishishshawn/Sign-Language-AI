import cv2
import socket
import pickle
import struct

# Define server address and port
server_address = 'localhost'
server_port = 8080

# Initialize socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_address, server_port))

# Access webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Serialize the frame
    data = pickle.dumps(frame)
    message_size = struct.pack("L", len(data))

    # Send the frame size and frame data
    client_socket.sendall(message_size + data)

    # Receive the prediction result
    result_size = struct.unpack("L", client_socket.recv(struct.calcsize("L")))[0]
    result_data = b""
    while len(result_data) < result_size:
        packet = client_socket.recv(4096)
        if not packet: break
        result_data += packet
    
    result = pickle.loads(result_data)

    # Overlay the result on the frame
    for (label, confidence, box) in result:
        (x, y, w, h) = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with the prediction
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client_socket.close()
