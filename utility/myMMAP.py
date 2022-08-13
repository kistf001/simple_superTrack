import mmap
from . import typeMaker

# server
def mmap_weight(process_number):
    g = []
    for index in range(process_number):
        EEEE = open("./mmapData/weight_"+str(index), "r+b")
        DDDD = mmap.mmap(EEEE.fileno(), 0)
        g.append(DDDD)
    return g
def mmap_learning(process_number):
    f = []
    for a in range(process_number):
        aaaa = open("./mmapData/process_mmap_"+str(a), "r+b")
        mm = mmap.mmap(aaaa.fileno(), 0)
        f.append(mm)
    return f
def mmap_get_actions(learning_data_mmap):
    mm = []
    for aw in learning_data_mmap:
        aw.seek(0)
        d = aw.read()
        vvv = typeMaker.tcp_binary_to_listed_numpy(d)
        mm.append(vvv)
    return mm 
def mmap_set_weight(mmap,weight_value):
    # 웨이트값 메모리에 적재
    for aw in mmap:
        bytes = weight_value
        bytes = typeMaker.listed_numpy_to_tcp_binary(bytes)
        aw.seek(0)
        aw.write(bytes)

# client
def weight_and_gen(index):
    # 웨이트
    with open("./mmapData/weight_"+str(index), "wb") as f:
        f.write(b"Hello Python!\n")
    EEEE = open("./mmapData/weight_"+str(index), "r+b")
    ssss = mmap.mmap(EEEE.fileno(), 0)
    ssss.resize(100*1000*1000)
    return ssss
def learning_and_gen(index):
    # 훈련 데이터
    with open("./mmapData/process_mmap_"+str(index), "wb") as f:
        f.write(b"Hello Python!\n")
    f =  open("./mmapData/process_mmap_"+str(index), "r+b")
    mm = mmap.mmap(f.fileno(), 0)
    mm.resize(100*1000*1000)
    return mm
def read_weight(mmap):
    mmap.seek(0)
    weight = mmap.read()
    return typeMaker.tcp_binary_to_listed_numpy(weight)
def write_learn(mmap,s):
    sdadfg = typeMaker.listed_numpy_to_tcp_binary(s)
    mmap.seek(0)
    mmap.write(sdadfg)