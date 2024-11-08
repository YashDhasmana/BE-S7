def crackSafe(n, k):
    seen = set()  # Set to keep track of seen sequences
    ans = []      # List to store the final sequence

    def dfs(node):
        for x in range(k):
            nei = node + str(x)  # Create a new sequence by appending x
            if nei not in seen:
                seen.add(nei)      # Mark this sequence as seen
                dfs(nei[1:])       # Recursively explore the next sequence
                ans.append(str(x)) # Add x to the final sequence

    dfs('0' * (n - 1))  # Start DFS with the initial node consisting of n-1 zeros
    return ''.join(ans) + '0' * (n - 1)  # Construct the final sequence

# Example usage
n = 2
k = 2
print(f"Cracked Safe Sequence: {crackSafe(n, k)}")
