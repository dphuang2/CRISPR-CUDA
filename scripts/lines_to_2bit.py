from to_2bit import four_bases_to_byte

with open("../src/sequences.txt") as fo:
  data = fo.read().split('\n')

with open("sequences.txt.translated", "wb") as fo:
    for i in range(len(data)):
        line = data[i]
        start, end = 0, 4
        translated = bytearray()
        while bool(line[start:end]):
            translated.append(four_bases_to_byte(*line[start:end]))
            start += 4
            end += 4
        fo.write(translated)
        if i < len(data) - 1:
            fo.write('\n')
