def circleGame(n, k):
    friends = list(range(1, n + 1))
    elimination = 0

    while len(friends) > 1:
        index = (elimination + k - 1) % len(friends)
        friends.pop(index)
    winner = friends[0]
    return winner

n = int(input("Enter the number of friends: "))
k = int(input("Enter the counting number: "))

winner = circleGame(n, k)
print(f"The Winner is Friend {winner}")
