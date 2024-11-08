def bin_packing(weights, capacity):
    # Step 1: Sort the items in decreasing order of their weights
    weights.sort(reverse=True)

    # Step 2: Initialize the list of bins, where each bin is a list of items
    bins = []

    # Step 3: Iterate over each item
    for weight in weights:
        placed = False

        # Try to place the item in the first bin that has enough remaining capacity
        for bin in bins:
            if sum(bin) + weight <= capacity:
                bin.append(weight)
                placed = True
                break

        # If no bin has enough capacity, create a new bin
        if not placed:
            bins.append([weight])

    # Output the number of bins used and the items in each bin
    return len(bins), bins


# Test the function
weights = [4, 8, 15, 10, 2, 1]
capacity = 9
bins_needed, bins = bin_packing(weights, capacity)
print(f"Minimum number of bins required: {bins_needed}")
for i, bin_items in enumerate(bins, 1):
    print(f"Bin {i}: {bin_items}")
