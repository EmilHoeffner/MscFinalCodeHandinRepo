'''
    Constructs a deterministic set of lst, where the order is based on the first appearance 
'''
def list_distinct(lst):
    distinct = []

    for elem in lst:
        if elem not in distinct:
            distinct.append(elem)

    return distinct



def main():
    lst1 = [1, 1, 2, 1, 3, 3, 2, 2, 4]
    lst2 = ["He is a", "unicorn", "He is a"]

    print(list_distinct(lst1))
    print(list_distinct(lst2))

if __name__ == "__main__":
    main()