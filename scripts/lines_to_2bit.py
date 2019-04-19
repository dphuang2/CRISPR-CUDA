from to_2bit import four_bases_to_byte
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process file of lines')
    parser.add_argument('file_path', help='file path')
    args = parser.parse_args()

    with open(args.file_path) as fo:
      data = fo.read().split('\n')

    with open("{}.twobit".format(args.file_path), "wb") as fo:
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
