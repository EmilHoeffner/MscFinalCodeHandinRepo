

'''
    outer: A list where inner is in it, e.g. [a,b,c,d,e]
    inner: A list inside outer, e.g. [b,c]

    The overlap indexes. In the case of the example [1, 2]
'''
def overlap_indexes(outer, inner):
    i = 0
    for i in range(len(outer)):
        out = outer[i]
        if out == inner[0]:
            indicies = [i]
            for inn in inner[1:]:
                i = i + 1
                if inn == outer[i]:
                    indicies.append(i)
                else:
                    break
            if len(indicies) == len(inner):
                return indicies
            else:
                continue
        else:
            i = i + 1
    
    return []



def main():
    outer = ["A", "Cat", "walked", "in", "Town"]
    inner = ["walked", "in"]

    print(overlap_indexes(outer, inner))

if __name__ == "__main__":
    main()