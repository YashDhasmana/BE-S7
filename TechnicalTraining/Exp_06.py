def compress_string(s):
    temp = []  # Temporary list to hold the compressed string
    i = 0
    len_s = len(s)

    while i < len_s:
        if i < len_s - 1 and s[i] == '3' and s[i + 1] == '3':
            temp.append('2')
            temp.append('3')
            i += 2  # Skip the next character as it is part of the pattern
        elif i < len_s - 2 and s[i] == '2' and s[i + 1] == '2' and s[i + 2] == '2':
            temp.append('3')
            temp.append('2')
            i += 3  # Skip the next two characters as they are part of the pattern
        elif s[i] == '5':
            temp.append('1')
            temp.append('5')
            i += 1
        elif s[i] == '1':
            temp.append('1')
            temp.append('1')
            i += 1
        else:
            temp.append(s[i])
            i += 1

    return ''.join(temp)


# original_string = "3322251"
original_string = "2223351"
print("Original string:", original_string)

compressed_string = compress_string(original_string)
print("Compressed string:", compressed_string)
