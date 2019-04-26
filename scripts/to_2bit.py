"""
This file is for mapping a file of one line (no new lines) of nucleotides to
twobit format (convert A, C, T, G to two bits)
"""

def four_bases_to_byte(*args):
    N = len(args)
    # T is used because it maps to 0b00
    one = args[0] if N > 0 else 'T'
    two = args[1] if N > 1 else 'T'
    three = args[2] if N > 2 else 'T'
    four = args[3] if N > 3 else 'T'
    return (four_bases_to_byte.mapping[one] << 6) + \
            (four_bases_to_byte.mapping[two] << 4) + \
            (four_bases_to_byte.mapping[three] << 2) + \
            (four_bases_to_byte.mapping[four])
four_bases_to_byte.mapping = {
        'T': 0b00,
        'C': 0b01,
        'A': 0b10,
        'G': 0b11
        }

if __name__ == "__main__":
    with open("./hg38.fa.no_newline.no_Ns.upper.no_Ns", "r") as fo:
        data = fo.read()

    i = 0
    translated = bytearray()
    while i < len(data) - 3:
        byte = four_bases_to_byte(data[i], data[i + 1], data[i + 2], data[i + 3])
        translated.append(byte)
        i += 4
        if i % 10000000 == 0:
            print("{} megabytes in".format(i / 1000000))

    # This code accounts for fact that the # of bases is not divisible by 4
    # by just replacing the missing bases with a byte with 0b00
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

    with open('hg38.twobit', 'wb') as fo:
        fo.write(translated)
