def makesquare(matchsticks):
    total_sum = sum(matchsticks)

    # If the total length isn't divisible by 4, we can't form a square
    if total_sum % 4 != 0:
        return False

    side_length = total_sum // 4
    matchsticks.sort(reverse=True)  # Sorting helps in optimization

    # Create an array to store the length of the 4 sides
    sides = [0] * 4
    # Create an array to store the matchsticks used for each side
    side_sticks = [[] for _ in range(4)]

    # Backtracking function to try forming sides
    def backtrack(index):
        if index == len(matchsticks):
            # If all matchsticks are used, check if all sides are equal to the target side length
            return sides[0] == sides[1] == sides[2] == sides[3] == side_length

        # Try to place the current matchstick in one of the four sides
        for i in range(4):
            if sides[i] + matchsticks[index] <= side_length:
                sides[i] += matchsticks[index]
                side_sticks[i].append(matchsticks[index])  # Add the matchstick to the current side

                if backtrack(index + 1):
                    return True

                # Backtrack if it didn't work
                sides[i] -= matchsticks[index]
                side_sticks[i].pop()  # Remove the last matchstick during backtrack

            # Optimization: if one side is still empty after attempting to place the matchstick,
            # don't try the other sides because they are in the same state
            if sides[i] == 0:
                break

        return False

    result = backtrack(0)
    print("Result:", result)  # Print the result first

    if result:
        print("\nThe matchsticks used are:")
        for i in range(4):
            if len(side_sticks[i]) == 1:
                print(f"Side {i + 1}: {side_sticks[i][0]} (1 stick)")
            else:
                sticks_str = " + ".join(map(str, side_sticks[i]))
                print(f"Side {i + 1}: {sticks_str} ({len(side_sticks[i])} sticks)")
        print("")
    return result

# Example
matchsticks = [2, 1, 3, 3, 3]
makesquare(matchsticks)  # Output will include the result and the sticks used for each side
