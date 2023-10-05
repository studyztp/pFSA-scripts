# Copyright (c) 2023 The Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from gem5.simulate.simulator import Simulator
from gem5.utils.requires import requires
from gem5.isas import ISA
import time
from gem5.simulate.exit_event import ExitEvent
from pathlib import Path
from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.memory import SingleChannelDDR3_1600
from gem5.prebuilt.riscvmatched.riscvmatched_board import RISCVMatchedBoard
from m5.objects import RedirectPath
import m5
import math
import json
from gem5.resources.workload import CustomWorkload
from gem5.resources.resource import BinaryResource, FileResource
from gem5.components.cachehierarchies.classic.no_cache import NoCache
from gem5.components.memory import SingleChannelDDR4_2400
from gem5.simulate.exit_event_generators import smarts_generator
import argparse
from subprocess import Popen

requires(isa_required=ISA.RISCV)

package_path = Path("/home/studyztp/test_ground/SPEC2006-package/package")

parser = argparse.ArgumentParser(
    description="A SPEC2006 run script."
)

benchmarks=[
    'povray',
    'tonto',
    'libquantum',
    'GemsFDTD',
    'sjeng',
    'hmmer',
    'bwaves',
    'mcf',
    'gromacs',
    'gcc',
    'lbm',
    'zeusmp',
    'bzip2',
    'calculix',
    'h264ref',
    'cactusADM',
    'gobmk',
    'namd',
    'astar',
    'leslie3d',
    'omnetpp',
    'perlbench',
    'milc'
]

parser.add_argument(
    "--benchmark",
    type=str,
    required=True,
    choices=benchmarks,
    help="The benchmark name.",
)

parser.add_argument(
    "--size",
    type=str,
    required=True,
    choices=["train","ref"],
    help="The benchmark size.",
)

parser.add_argument(
    "--input-id",
    type=str,
    required=True,
    help="The benchmark input id.",
)

parser.add_argument(
    "--json-store-path",
    type=str,
    required=True,
    help="where to save the json file"
)

parser.add_argument(
    "--cpts-store-path",
    type=str,
    required=True,
    help="where to save the cpts"
)

parser.add_argument(
    "--output-store-path",
    type=str,
    required=True,
    help="where to save the output"
)

parser.add_argument(
    "--proc-limit",
    type=int,
    required=True,
    help="limitation of number of checkpoints restored at a time"
)

parser.add_argument(
    "--proc-over-allowance",
    type=int,
    required=True,
    help="keep making how many checkpoints when proc limit is reached"
)

parser.add_argument(
    "--gem5-binary",
    type=str,
    required=True,
    help="The gem5 binary.",
)

parser.add_argument(
    "--child-script",
    type=str,
    required=True,
    help="The child script.",
)

parser.add_argument(
    "-U",
    type=int,
    required=True,
    help="The instruction length of each sample unit.",
)

parser.add_argument(
    "-n",
    type=int,
    required=True,
    help="The number of simulation intervals.",
)

args = parser.parse_args()

with open(Path("/home/studyztp/test_ground/SPEC2006-package/spec2006-simpoint-workloads.json").as_posix()) as file:
    spec2006_simpoint_resources_info = json.load(file)

with open(Path("/home/studyztp/test_ground/experiments/SMART/spec2006-inst.json").as_posix()) as file:
    spec2006_inst_info = json.load(file)

json_file_path = Path(args.json_store_path)

info_json = {}

cpt_store_path = Path(args.cpts_store_path)
output_store_path = Path(args.output_store_path)
child_script = Path(args.child_script)

n = args.n
U = args.U

dynmaic_insts = int(spec2006_inst_info[args.size][args.benchmark][args.input_id])

N = int(math.ceil(dynmaic_insts/U))
k = int(math.ceil(N/n)) 
if int(math.ceil(k/2)) > 100000:
    W = 100000*U
else:
    W = int(int(math.ceil(k/2))*U)

warmup_start = (k-1)*U - W
detailed_end = k*U

info_json["k"] = k
info_json["U"] = U
info_json["W"] = W
info_json["N"] = N
info_json["total dynamic insts"] = dynmaic_insts

binary_info = spec2006_simpoint_resources_info[f"binary-{args.benchmark}-{args.size}-input-{args.input_id}"]
binary_name = binary_info["binary"]
stdin = binary_info["stdin"]
if stdin != None:
    stdin = Path(package_path/f"{args.benchmark}/input/{args.size}/{stdin}")
    stdin = FileResource(local_path=stdin.as_posix())
stdout = binary_info["stdout"]
if stdout != None:
    stdout = Path(stdout)
stderr = binary_info["stderr"]
if stderr != None:
    stderr = Path(stderr)
binary = BinaryResource(local_path=Path(package_path/f"{args.benchmark}/exe/{binary_name}").as_posix())

SPEC2006_workload = CustomWorkload(
    function = "set_se_binary_workload",
    parameters = {
        "binary" : binary,
        "arguments" : binary_info["arguments"],
        "stdin_file": stdin,
        "stderr_file": stderr,
        "stdout_file": stdout
    },
)

processor = SimpleProcessor(
    cpu_type=CPUTypes.ATOMIC, isa=ISA.RISCV, num_cores=1
)

cache_hierarchy = NoCache()

memory = SingleChannelDDR4_2400("16GB")

board = SimpleBoard(
    clk_freq="1.2GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)

board.set_workload(SPEC2006_workload)

print(f"redirect {Path.cwd().absolute()} to {package_path}/{args.benchmark}/input/{args.size}/\n")
board.redirect_paths = [RedirectPath(app_path=f".", host_paths=[f"{package_path}/{args.benchmark}/input/{args.size}/"])]

cpt_info = {}

def exit_event_handler():
    global cpt_info
    cpt_counter = 0
    cpt_waitlist = []
    while True:
        print("get to warmup start, now take cpt\n")
        print(f"simulated {processor.get_cores()[0].core.totalInsts()}\n")
        cpt_path = Path(cpt_store_path/f"{cpt_counter}")
        m5.checkpoint(cpt_path.as_posix())
        cpt_counter += 1
        cpt_waitlist.append(cpt_path)
        while len([i for i in cpt_store_path.iterdir()]) < args.proc_limit and len(cpt_waitlist) > 0:
            target_cpt = cpt_waitlist.pop()
            target_store_path = Path(output_store_path/f"{target_cpt.name}")
            print(f"start detailed execution for {target_cpt.as_posix()}\n")
            command = [
                f"{args.gem5_binary}",
                 "-re",
                 f"--outdir={target_store_path.as_posix()}", 
                 f"{child_script}",
                 f"--benchmark={args.benchmark}",
                 f"--size={args.size}",
                 f"--input-id={args.input_id}",
                 f"--cpt-path={target_cpt.as_posix()}",
                 f"-U={U}",
                 f"-W={W}"]
            print("ran\n")
            print(command)
            print("\n")
            
            proc = Popen(command)
            print(f"pid is \n{proc.pid}\n")
            cpt_info[f"{target_cpt.name}"] = {
                "curTick":m5.curTick(),
                "pid":proc.pid,
                "command":command
            }
        while len([i for i in cpt_store_path.iterdir()]) > args.proc_limit:
            if len(cpt_waitlist) < args.proc_over_allowance:
                break
            time.sleep(10*60)

        print(f"schedule exit event after {U} + {W} = {U+W}\n")
        processor.get_cores()[0].core.scheduleInstStopAnyThread(U+W)
        print("fall back to simulation\n")
        yield False
        print("get to end of simulation interval\n")
        print(f"simulated {processor.get_cores()[0].core.totalInsts()}\n")

        print(f"schedule exit event after {warmup_start}\n")
        processor.get_cores()[0].core.scheduleInstStopAnyThread(warmup_start)

        print("fall back to simulation\n")
        yield False


simulator = Simulator(
    board=board,
    on_exit_event={
        ExitEvent.MAX_INSTS: exit_event_handler()
    }
)

print(f"{args.benchmark} with {args.input_id} input has {dynmaic_insts} instructions\nW is {W}\nU is {U}\nk is {k}\nN is {N}\n")

print(f"schedule exit event after {warmup_start}\n")
processor.get_cores()[0].core.max_insts_any_thread(warmup_start)

print("starting simulation\n")

start_time = time.time()

simulator.run()

print("made all cpts")

info_json["cpt-info"] = cpt_info

while len([i for i in cpt_store_path.iterdir()]) > 0:
    time.sleep(10*60)

time_spent = time.time() - start_time

print(time_spent)

info_json["host time"] = time_spent

with open(json_file_path.as_posix(),"w") as file:
    json.dump(info_json,file,indent=4)

print("done running \n")
