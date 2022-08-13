import multiprocessing as mp
import m_client, m_server
import setup

if __name__ == "__main__":

    process_number = 1

    if(process_number):

        mutex0 = [ mp.Lock() for d in range(process_number) ]
        mutex1 = [ mp.Lock() for d in range(process_number) ]

        process = [ mp.Process(
            target=m_client.process,
            args=(
                number,
                mutex0[number],
                mutex1[number],
                int(4096/process_number),
            )
        ) for number in range(process_number) ]

        [ p.start() for p in process ]

    m_server.process(process_number,mutex0,mutex1)
        