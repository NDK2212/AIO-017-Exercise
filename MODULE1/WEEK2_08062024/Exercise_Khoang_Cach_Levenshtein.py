if __name__ == "__main__":
    source = input("Enter your source string:")
    target = input("Enter your target string:")
    source_size = len(source)
    target_size = len(target)
    matrix = [[0 for num in range(0, target_size + 1)]
              for _ in range(0, source_size + 1)]
    for i in range(0, target_size+1):
        matrix[0][i] = i
    for i in range(0, source_size+1):
        matrix[i][0] = i
    for i in range(1, source_size+1):
        for j in range(1, target_size+1):
            sub_cost = 0 if source[i-1] == target[j-1] else 1
            matrix[i][j] = min(matrix[i-1][j] + 1, matrix[i]
                               [j-1]+1, matrix[i-1][j-1] + sub_cost)
    print(matrix[source_size][target_size])
