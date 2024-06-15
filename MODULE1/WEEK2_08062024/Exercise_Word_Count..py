def word_count(file_path):
    my_file = open(file_path, 'r')
    my_data = my_file.readlines()
    my_file.close()
    my_dict = {}
    for row in my_data:
        current_line = row.split()
        for i in current_line:
            word = i.lower()
            if word not in my_dict.keys():
                my_dict[word] = 1
            else:
                my_dict[word] += 1
    list_of_keys = list(my_dict.keys())
    list_of_keys.sort()
    list_of_keys = {i: my_dict[i] for i in list_of_keys}
    print(list_of_keys)


if __name__ == "__main__":
    word_count('WEEK2_08062024/Word_Count_data.txt')
