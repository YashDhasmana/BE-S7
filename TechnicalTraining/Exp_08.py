def calculateMinimumHP(dungeon):
    if not dungeon:
        return 0

    m, n = len(dungeon), len(dungeon[0])

    # Initialize a DP table with high values (m+1, n+1 to avoid boundary checks)
    dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
    direction = [[''] * n for _ in range(m)]  # To track the direction taken (R or D)

    # The knight needs at least 1 health point to survive at the bottom-right corner
    dp[m][n-1] = dp[m-1][n] = 1

    # Print the initial dungeon state
    print("\nInitial dungeon state:")
    for row in dungeon:
        print(row)
    print()

    # Traverse the dungeon in reverse (from bottom-right to top-left)
    for i in range(m-1, -1, -1):
        for j in range(n-1, -1, -1):
            # Calculate minimum health needed from right or down cells
            if dp[i+1][j] < dp[i][j+1]:
                min_health_needed = dp[i+1][j] - dungeon[i][j]
                direction[i][j] = 'D'  # Move down
            else:
                min_health_needed = dp[i][j+1] - dungeon[i][j]
                direction[i][j] = 'R'  # Move right

            dp[i][j] = max(1, min_health_needed)

    # The top-left corner will have the minimum initial health required
    min_initial_health = dp[0][0]

    # Reconstruct the path from (0, 0) to (m-1, n-1)
    path = []
    i, j = 0, 0
    while i < m-1 or j < n-1:
        path.append((i, j))
        if direction[i][j] == 'R':
            j += 1
        else:
            i += 1
    path.append((m-1, n-1))  # Add the bottom-right corner

    # Print the path
    print("Path taken:")
    for p in path:
        print(f"Room {p}: {dungeon[p[0]][p[1]]}")

    return min_initial_health

# Test cases
dungeon1 = [[30, -3, 3], [65, -10, 1], [10, -90, -5]]

print("\nMinimum initial health required:", calculateMinimumHP(dungeon1))  # Output: 7
print("\n")
