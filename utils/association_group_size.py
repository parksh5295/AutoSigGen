# Functions for counting the number of features in a related group
# Need to double-check input, output to make sure it can be used when counting dictionary keys

def association_group_size(related_groups):
    # Initializing a dictionary to count by number of elements in a group
    count_by_size = {}

    # Count elements for each group in related_groups
    for group in related_groups:
        size = len(group)  # Number of elements in group
        if size in count_by_size:
            count_by_size[size] += 1
        else:
            count_by_size[size] = 1

    # Result output
    sorted_counts = sorted(count_by_size.items())

    # Output
    for size, count in sorted_counts:
        print(f"Element count {size}: {count} groups")