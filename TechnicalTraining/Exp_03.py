class Node:
    def __init__(self, val, prev=None, next=None, child=None):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child

def flatten(head):
    if not head:
        return head

    # Stack to keep track of nodes
    stack = []
    current = head

    while current:
        if current.child:
            # If the node has a child, push the next node onto the stack
            if current.next:
                stack.append(current.next)

            # Point the current node's next to the child
            current.next = current.child
            current.child.prev = current
            # Remove the child pointer
            current.child = None

        if not current.next and stack:
            # If no next node but stack is not empty, pop from stack and continue
            next_node = stack.pop()
            current.next = next_node
            next_node.prev = current

        current = current.next

    return head


# Creating the multilevel doubly linked list with the specified values
head = Node(1)
node2 = Node(2)
node3 = Node(3)
node4 = Node(4)
node5 = Node(5)
node6 = Node(6)
child1 = Node(7)
child2 = Node(8)

head.next = node2
node2.prev = head
node2.next = node3
node3.prev = node2
node3.next = node4
node4.prev = node3
node4.next = node5
node5.prev = node4
node5.next = node6
node6.prev = node5

node3.child = child1
child1.next = child2
child2.prev = child1

# Flattening the list
flatten(head)
# Printing flattened list
current = head
print("Flattened List: ")
while current:
    print(current.val, end=" ")
    current = current.next
