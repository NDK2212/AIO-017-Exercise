if __name__ == "__main__":
    num_list = [3, 2, 5, 2, -2, 7, -7, 1, 9, 7, 13, 3, 2]
    k = int(input("Enter the size of your sliding window:"))
    sz = len(num_list)
    while k == 0 or k > sz:
        k = int(input(
            f"Your input must be strictly larger than 0 and smaller than {sz}. Reenter the size of your sliding window:"))
    for i in range(0, sz - k + 1):
        max_value = num_list[i]
        for j in range(1, k):
            max_value = max(max_value, num_list[i+j])
        print(f"Maximum value of the {i+1}th sliding window is {max_value}.")
