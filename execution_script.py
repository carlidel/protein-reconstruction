import multiprocessing
import subprocess
import shlex
import os
import sys

from multiprocessing.pool import ThreadPool


def call_proc(cmd):
    p = subprocess.Popen(shlex.split(
        cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    print(out, file=open('progress.txt', 'a'))
    print(err, file=open('errors.txt', 'a'))
    return (out, err)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        CPUS = int(sys.argv[1])
    else:
        CPUS = multiprocessing.cpu_count()

    if os.path.exists('progress.txt'):
        os.remove('progress.txt')
    if os.path.exists('errors.txt'):
        os.remove('errors.txt')

    pool = ThreadPool(CPUS)
    results = []
    with open("processed_pdb/names.txt", 'r') as f:
        names = [name[:-1] for name in f]
    arg_names = [[] for i in range(CPUS)]
    for i, name in enumerate(names):
        arg_names[i % CPUS].append(name)
    
    separator = " "
    for arg in arg_names:
        name = separator.join(arg)
        print("python new_main.py " + name)
        results.append(pool.apply_async(
            call_proc, args=("python new_main.py " + name,)))

    pool.close()
    pool.join()
    for i, result in enumerate(results):
        out, err = result.get()
        print("err{}: {}".format(i, err))
