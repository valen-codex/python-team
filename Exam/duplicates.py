# Function to remove duplicates from a list while preserving order without using set()

my_list= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 4, 5, 6, 11, 12, 13, 14, 15, 2, 3, 4]
# def remove_duplicates(input_list):
    # new_list = []
    # for item in input_list:
    #     if item not in new_list:
    #         new_list.append(item)
    # return new_list
new_list = []
for item in my_list:
    if item not in new_list:
        new_list.append(item)
print(new_list)
