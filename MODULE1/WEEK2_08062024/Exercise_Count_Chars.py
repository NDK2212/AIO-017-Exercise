def count_chars(s):
    my_dict = {}
    for c in s:
        if c not in my_dict.keys():
            my_dict[c] = 1
        else:
            my_dict[c] += 1
    return my_dict


if __name__ == "__main__":
    s = input("Enter a string: ")
    print(count_chars(s))
