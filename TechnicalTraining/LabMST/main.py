class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def maxPathSumHelper(node, max_sum):
    if node is None:
        return 0

    # Recursively find the maximum path sum of the left and right subtrees
    left = maxPathSumHelper(node.left, max_sum)
    right = maxPathSumHelper(node.right, max_sum)

    # Maximum path sum with the current node as the highest node in the path
    max_single_path = max(max(left, right) + node.val, node.val)

    # Maximum path sum with the current node as the highest node, including both subtrees
    max_top_path = max(max_single_path, left + right + node.val)

    # Update the global maximum path sum
    max_sum[0] = max(max_sum[0], max_top_path)

    return max_single_path

def maxPathSum(root):
    max_sum = [float('-inf')]  # Initialize the global maximum path sum
    maxPathSumHelper(root, max_sum)
    return max_sum[0]

# Example usage
if __name__ == "__main__":
    # Create the binary tree
    root = TreeNode(-10)
    root.left = TreeNode(9)
    root.right = TreeNode(20)
    root.right.left = TreeNode(15)
    root.right.right = TreeNode(7)

    # Find the maximum path sum
    result = maxPathSum(root)
    print(f"Maximum path sum: {result}")
