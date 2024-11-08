def print_pascals_triangle(n):
    # Initialize the top of Pascal's Triangle
    triangle = [[1]]

    # Generate the rest of the rows
    for i in range(1, n):
        row = [1]  # Start with 1
        for j in range(1, i):
            row.append(triangle[i - 1][j - 1] + triangle[i - 1][j])
        row.append(1)  # End with 1
        triangle.append(row)

    # Print the triangle
    for row in triangle:
        print(row)

# User input for number of rows
n = int(input("Enter the number of rows: "))
print_pascals_triangle(n)
