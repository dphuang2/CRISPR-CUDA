import pdb
translated = bytearray()

with open("./hg38.fa.no_newline.no_Ns.upper.no_Ns", "r") as fo:
    data = fo.read()

# http://genome.ucsc.edu/FAQ/FAQformat.html#format7
mapping = {
        'T': 0b00,
        'C': 0b01,
        'A': 0b10,
        'G': 0b11
        }

i = 0
while i < len(data) - 3:
    byte = (mapping[data[i]] << 6) + \
            (mapping[data[i + 1]] << 4) + \
            (mapping[data[i + 2]] << 2) + \
            (mapping[data[i + 3]])
    translated.append(byte)
    i += 4
    if i % 10000000 == 0:
        print("{} megabytes in".format(i / 1000000))

with open('hg38.2bit.processed.phase1', 'wb') as fo:
    fo.write(translated)

if len(data) - i == 3:
    byte = (mapping[data[i]] << 6) + \
            (mapping[data[i + 1]] << 4) + \
            (mapping[data[i + 2]] << 2)
elif len(data) - i == 2:
    byte = (mapping[data[i]] << 6) + \
            (mapping[data[i + 1]] << 4)
elif len(data) - i == 1:
    byte = (mapping[data[i]] << 6)
translated.append(byte)

with open('hg38.2bit.processed.phase2', 'wb') as fo:
    fo.write(translated)
