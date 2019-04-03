import multiprocessing
import subprocess
import shlex

from multiprocessing.pool import ThreadPool


def call_proc(cmd):
    p = subprocess.Popen(shlex.split(
        cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    print(out, file=open('progress.txt', 'a'))
    return (out, err)


if __name__ == "__main__":
    pool = ThreadPool(multiprocessing.cpu_count())
    results = []
    with open("processed_pdb/names.txt", 'r') as f:
        names = [name[:-1] for name in f]
    for name in names:
        print("python new_main.py " + name)
        results.append(pool.apply_async(
            call_proc, args=("python new_main.py " + name,)))

    pool.close()
    pool.join()
    for i, result in enumerate(results):
        out, err = result.get()
        print("err{}: {}".format(i, err))
