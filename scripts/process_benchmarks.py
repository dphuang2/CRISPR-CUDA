import os
from pprint import pprint
import np
import pdb
import re
from collections import defaultdict

def escape_ansi(line):
    ansi_escape =re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', line)

folder_path = "../data"
file_paths = [os.path.join(folder_path, fp) for fp in os.listdir(folder_path) if "benchmark" in fp]
print(file_paths)

benchmarks = []
for file_path in file_paths:
    with open(file_path, "r") as fp:
        data = fp.read().split('\n')
        data = [escape_ansi(line) for line in data if "TIMER" in line or "Info" in line]
        benchmarks.append(data)

csv = defaultdict(lambda: defaultdict(list))
current_test_length = 0
test_lengths = set()
for benchmark in benchmarks:
    for line in benchmark:
        if "test_length" in line:
            pdb.set_trace()
            current_test_length = int(line.split("test_length")[-1].split()[-1])
            test_lengths.add(current_test_length)
        elif "Executing" in line and "took" in line:
            pdb.set_trace()
            ms = float(line.split("took")[-1].split("ms")[0])
            method = line.split("took")[0].split(" ", 1)[-1].strip()
            csv[method][current_test_length].append(ms)

del csv["Naive CPU"]
pprint(test_lengths)
pprint(csv.keys())
with open("benchmarks.csv", "w") as fp:
    fp.write('Methods,')
    for length in sorted(list(test_lengths)):
        fp.write(str(length) + ',')
    fp.write('\n')
    for method, lengths in csv.items():
        fp.write(method + ",")
        for length in sorted(lengths.keys()):
            iterations = lengths[length]
            fp.write(str(np.mean(iterations)) + ',')
        fp.write('\n')
