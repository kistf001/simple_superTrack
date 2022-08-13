import numpy as np
import socket
import setup

def socket_init_client():
    ClientSocket = socket.socket()
    try:
        ClientSocket.connect((setup.ip, setup.port))
    except socket.error as e:
        print(str(e))
    return ClientSocket
def socket_init_server():
    ServerSocket = socket.socket()
    try:
        ServerSocket.bind((setup.ip, setup.port))
    except socket.error as e:
        print(str(e))
    return ServerSocket

def listed_numpy_to_tcp_binary(state_datas):
    #
    aa = b""
    #
    package_size = np.array(len(state_datas),dtype=np.int32).tobytes()
    aa += package_size
    #
    for state_data in state_datas:
        # 무조코 돌린 결과를 (바이트)와 (어레이형태)로 저장한다.
        state_data = np.array(state_data,dtype=np.float32)
        state_data_shape = np.array(state_data.shape,dtype=np.int32)
        state_data_shape_dimension = np.array(state_data_shape.shape,dtype=np.int32)
        # 바이트로
        aa += state_data_shape_dimension.tobytes()
        aa += state_data_shape.tobytes()
        aa += state_data.tobytes()
    #
    return aa
def tcp_binary_to_listed_numpy(data):

    def read_binary(start_p, range_p,dtype=np.int32):
        return np.frombuffer(data[start_p:start_p+range_p],dtype=dtype)

    dfdf = []

    ## 리스트로 묶인 개수
    now_p = 0
    shape_p_r = 4
    data1 = read_binary(now_p, shape_p_r)[0]

    for a in range(0,data1):

        ## 넘파이 베열의 차원
        now_p += shape_p_r
        shape_p_r = 4
        data1 = read_binary(now_p, shape_p_r)[0]

        # 넘파이 베열의 형태
        now_p += shape_p_r
        shape_p_r = 4*data1
        data2 = read_binary(now_p, shape_p_r)
        shape = data2

        # 넘파이 베열
        now_p += shape_p_r
        shape_p_r = 4
        for a in data2:
            shape_p_r *= a
        data1 = read_binary(now_p, shape_p_r,dtype=np.float32)

        dfdf.append(data1.reshape(tuple(shape)))

    return dfdf

def listed_numpy_to_tcp_binary_64(state_datas):
    #
    aa = b""
    #
    package_size = np.array(len(state_datas),dtype=np.int64).tobytes()
    aa += package_size
    #
    for state_data in state_datas:
        # 무조코 돌린 결과를 (바이트)와 (어레이형태)로 저장한다.
        state_data = np.array(state_data,dtype=np.float64)
        state_data_shape = np.array(state_data.shape,dtype=np.int64)
        state_data_shape_dimension = np.array(state_data_shape.shape,dtype=np.int64)
        # 바이트로
        aa += state_data_shape_dimension.tobytes()
        aa += state_data_shape.tobytes()
        aa += state_data.tobytes()
    #
    return aa
def tcp_binary_to_listed_numpy_64(data):

    def read_binary(start_p, range_p,dtype=np.int64):
        return np.frombuffer(data[start_p:start_p+range_p],dtype=dtype)

    dfdf = []

    ## 리스트로 묶인 개수
    now_p = 0
    shape_p_r = 8
    data1 = read_binary(now_p, shape_p_r)[0]

    for a in range(0,data1):

        ## 넘파이 베열의 차원
        now_p += shape_p_r
        shape_p_r = 8
        data1 = read_binary(now_p, shape_p_r)[0]

        # 넘파이 베열의 형태
        now_p += shape_p_r
        shape_p_r = 8*data1
        data2 = read_binary(now_p, shape_p_r)
        shape = data2

        # 넘파이 베열
        now_p += shape_p_r
        shape_p_r = 8
        for a in data2:
            shape_p_r *= a
        data1 = read_binary(now_p, shape_p_r,dtype=np.float64)

        dfdf.append(data1.reshape(tuple(shape)))

    return dfdf

def tcp_read(connection):
    # 데이터 길이를 받아낸다.
    MSGLEN = 8
    chunks = []
    bytes_recd = 0
    while bytes_recd < MSGLEN:
        chunk = connection.recv(MSGLEN)
        if chunk == b'':
            raise RuntimeError("socket connection broken")
        chunks.append(chunk)
        bytes_recd = bytes_recd + len(chunk)
    payload_size = np.frombuffer( b''.join(chunks), dtype=np.int64 )[0]
    # 데이터를 받음
    MSGLEN = payload_size
    chunks = []
    bytes_recd = 0
    while bytes_recd < MSGLEN:
        chunk = connection.recv(4096)
        if chunk == b'':
            raise RuntimeError("socket connection broken")
        chunks.append(chunk)
        bytes_recd = bytes_recd + len(chunk)
    return b''.join(chunks)
def tcp_send(ClientSocket,msg):
    # EOT가 없는 소켓을 위해 데이터 길이를 전송한다.
    payload=np.array(len(msg),dtype=np.int64).tobytes()
    MSGLEN = 8
    totalsent = 0
    while totalsent < MSGLEN:
        sent = ClientSocket.send(payload[totalsent:])
        if sent == 0:
            raise RuntimeError("socket connection broken")
        totalsent = totalsent + sent
    # 페이로드를 전송한다.
    MSGLEN = len(msg)
    totalsent = 0
    while totalsent < MSGLEN:
        sent = ClientSocket.send(msg[totalsent:])
        if sent == 0:
            raise RuntimeError("socket connection broken")
        totalsent = totalsent + sent

#################################################

def data_exchange_client(a,b):
    tcp_send(a,b"1111")
    v_param = tcp_read(a)
    tcp_send(a,b"1111")
    p_param = tcp_read(a)
    tcp_send(a,b)
    return v_param, p_param
def data_exchange_server(a,b,c):
    Response = tcp_read(a)
    tcp_send(a,b)
    Response = tcp_read(a)
    tcp_send(a,c)
    Response = tcp_read(a)
    return Response

def server_close():
    a=1
def client_close():
    aa=1

if __name__ == "__main__":
    a = [np.array([3.,2.,1.,5.,3.0],dtype=np.float64)]
    a = listed_numpy_to_tcp_binary_64(a)
    a = tcp_binary_to_listed_numpy_64(a)
    print(a[0].dtype)