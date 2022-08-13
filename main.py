from algorithm import agent1 as agent
agent.torch.set_num_threads(8)

A = agent.Agent()

for dd in range(0,100000):

    A.run_data()
    
    if(dd %5)==0:
        A.param_export()