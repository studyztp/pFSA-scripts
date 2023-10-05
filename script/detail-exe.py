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
from gem5.simulate.exit_event import ExitEvent
from pathlib import Path
from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.memory import SingleChannelDDR3_1600
from gem5.prebuilt.riscvmatched.riscvmatched_board import RISCVMatchedBoard
from m5.objects import RedirectPath
import m5
import shutil
import json
from gem5.resources.workload import CustomWorkload
from gem5.resources.resource import BinaryResource, FileResource
import argparse

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
    "--cpt-path",
    type=str,
    required=True,
    help="The benchmark input id.",
)

parser.add_argument(
    "-U",
    type=int,
    required=True,
    help="The instruction length of each sample unit.",
)

parser.add_argument(
    "-W",
    type=int,
    required=True,
    help="The instruction length of the warmup interval.",
)

args = parser.parse_args()

with open(Path("/home/studyztp/test_ground/SPEC2006-package/spec2006-simpoint-workloads.json").as_posix()) as file:
    spec2006_simpoint_resources_info = json.load(file)

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
        "stdout_file": stdout,
        "checkpoint" : Path(args.cpt_path)
    },
)

board = RISCVMatchedBoard(
    clk_freq="1.2GHz",
    l2_size="2MB"
)

board.set_workload(SPEC2006_workload)

print(f"redirect {Path.cwd().absolute()} to {package_path}/{args.benchmark}/input/{args.size}/\n")
board.redirect_paths = [RedirectPath(app_path=f".", host_paths=[f"{package_path}/{args.benchmark}/input/{args.size}/"])]

def exit_event_handler():
    current_inst = board.get_processor().get_cores()[0].core.totalInsts()
    print(f"simulated {current_inst} instructions\n")
    print("dump and reset stats\n")
    m5.stats.dump()
    m5.stats.reset()
    print(f"schedule MAX_INSTS exit events at {args.U} instructions")
    board.get_processor().get_cores()[0].core.scheduleInstStopAnyThread(args.U)
    print("fall back to simulation\n")
    yield False
    print(f"get to detailed end after simulating {board.get_processor().get_cores()[0].core.totalInsts()-current_inst} instructions\n")
    print("now dump stats and exit simulation\n")
    m5.stats.dump()
    yield True

simulator = Simulator(
    board=board,
    on_exit_event={
        ExitEvent.MAX_INSTS: exit_event_handler()
    }
)

warmup_end = args.W

print(f"schedule MAX_INSTS exit events at {warmup_end} instructions")
board.get_processor().get_cores()[0].core.max_insts_any_thread(warmup_end)

print("starting simulation\n")

simulator.run()

print(f"now remove cpt at {args.cpt_path}")

shutil.rmtree(Path(args.cpt_path))

print("done running \n")
