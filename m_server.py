import time
import numpy
from utility import myMMAP
from algorithm import agent1 as agent
from algorithm import agent2 

def process(process_number,lock_0,lock_1):

    start_ppo = time.time()

    agent.torch.set_num_threads(8)

    # 클라이언트가 준비될 떄 까지 대기
    time.sleep(2)

    A = agent.Agent()

    # 웨이트, 훈련 데이터 통
    network_weight_mmap = myMMAP.mmap_weight(process_number)
    learning_data_mmap = myMMAP.mmap_learning(process_number)

    #========================== 메인루프 시작 ============================
    for dd in range(1,int(1e+15)):
        
        # ============================ ============================
        # 웨이트값 메모리에 적재
        parameter = A.param_get()
        myMMAP.mmap_set_weight( network_weight_mmap, parameter )
        # ============================ ============================


        # ============================ ============================
        # 서버 트리거
        [ d.release() for d in lock_0 ]
        time.sleep(0.1)
        print("==================== start ======================")
        [ d.acquire() for d in lock_1 ]         # 클라이언트들이 끝날때까지 잠김
        print("================== client end ===================")
        # ============================ ============================


        # ============================ ============================
        # 훈련 데이터
        _buffer_data = myMMAP.mmap_get_actions(learning_data_mmap)
        #_buffer_data = mm
        #[
        #    numpy.concatenate( [ a[0] for a in mm ] ),
        #    numpy.concatenate( [ a[1] for a in mm ] ),
        #    numpy.concatenate( [ a[2] for a in mm ] ),
        #    numpy.concatenate( [ a[3] for a in mm ] ),
        #    numpy.concatenate( [ a[4] for a in mm ] ),
        #    [ a[5] for a in mm ] ,
        #    [ a[6] for a in mm ] ,
        #]
        #a = numpy.array(_buffer_data[-2]).mean()
        #b = numpy.array(_buffer_data[-1]).mean()
        #print( dd, a, b,len(_buffer_data[1]) )
        A.run_learning(_buffer_data)
        if(dd %5)==0:
            A.param_export()
        if(dd %100)==0:
            print(time.time()-start_ppo)
        # ============================ ============================


        # ============================ ============================
        print("==================== end  ======================")
        print()
        print()
        # ============================ ============================


if __name__=="__main__":
    print(numpy.concatenate([[0],[0],[0],]))