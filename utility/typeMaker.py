import numpy as np

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
