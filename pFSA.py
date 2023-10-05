from pathlib import Path
import json
from multiprocessing import Pool
import subprocess

# benchmarks = [
# 'povray',
# 'tonto',
# 'libquantum',
# 'GemsFDTD',
# 'sjeng',
# 'hmmer',
# 'bwaves',
# 'mcf',
# 'gromacs',
# 'gcc',
# 'lbm',
# 'zeusmp',
# 'bzip2',
# 'calculix',
# 'h264ref',
# 'cactusADM',
# 'gobmk',
# 'namd',
# 'astar',
# 'leslie3d',
# 'omnetpp',
# 'perlbench',
# 'milc'
# ]

gem5_binary = Path("/home/studyztp/test_ground/studyztp/SMART/build/NUGGET_RISCV_MI/gem5.fast")
config_script = Path("/home/studyztp/test_ground/experiments/pFSA/script/main.py")
child_script = Path("/home/studyztp/test_ground/experiments/pFSA/script/detail-exe.py")
output_path = Path("/home/studyztp/test_ground/experiments/pFSA/output")

def create_processs(run):
    command = run["command"]
    print("running:")
    for cmd in command:
        print(cmd)
    print("\n")
    process_info = subprocess.run(command)
    return process_info


if __name__ == "__main__":

    # with open("/home/studyztp/test_ground/experiments/SMART/spec2006-inst.json") as file:
    #     benchmarks_data = json.load(file)

    benchmarks_data = {
        "train": {
            "perlbench" : [2]
        }
    }

    all_runs = []
    for size, size_list in benchmarks_data.items():
        for bench, input_list in size_list.items():
            for input in input_list:
                run_output_path = Path(output_path/f"{bench}-{size}-{input}")

                run_output_path.mkdir()

                run_cpt_store_path = Path(run_output_path/"cpt")
                run_m5_path = Path(run_output_path/"m5")
                run_all_restore = Path(run_output_path/"restore")

                run_cpt_store_path.mkdir()
                run_all_restore.mkdir()

                json_output_path = Path(run_m5_path/"smarts_stats.json")
                
                gem5_command = ["-re", f"--outdir={run_m5_path.as_posix()}"]
                script_command = [f"--benchmark={bench}", f"--input-id={input}", f"--size={size}",f"--json-store-path={json_output_path.as_posix()}",
                                  f"--cpts-store-path={run_cpt_store_path.as_posix()}", f"--output-store-path={run_all_restore.as_posix()}",
                                  "--proc-limit=15", "--proc-over-allowance=5", f"--gem5-binary={gem5_binary.as_posix()}",
                                  f"--child-script={child_script.as_posix()}", "-U=1000", "-n=10000"]
                command = [gem5_binary.as_posix()] + gem5_command + [config_script.as_posix()] + script_command
                runball = {
                    "command":command
                }
                all_runs.append(runball)


    with Pool() as pool:
        pool.map(create_processs,all_runs)


