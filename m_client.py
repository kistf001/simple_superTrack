import time
import numpy
from utility import myMMAP
from algorithm import agent1 as agent
from algorithm import agent2 

def process(index,lock_0,lock_1,samples):

    agent.torch.set_num_threads(1)
    
    A = agent.Agent(samples)

    ssss = myMMAP.weight_and_gen(index)
    mmap_exp_data = myMMAP.learning_and_gen(index)

    lock_0.acquire(block=False)

    start = 0

    # Don't forget to set the random seed value differently.
    # Some algorithms have error by the same random seed value.
    agent.torch.manual_seed(index*-100000)
    agent.env.numpy.random.seed(index*100000)
    
    for dd in range(1,int(1e+15)):
        
        lock_0.acquire()                # 서버가 신호를 주면 풀림
        lock_0.acquire(block=False)     # 다음을 위해 잠금
        lock_1.acquire(block=False)     # for server 

        # 웨이트
        #weight = myMMAP.read_weight(ssss)
        #A.param_set(weight)
        
        experience_data = A.run_data()

        # 훈련 데이터
        myMMAP.write_learn(mmap_exp_data,experience_data)

        #########################################################
        print(index, dd, time.time()-start, )
        start = time.time()
        #########################################################

        # server lock release
        lock_1.release()