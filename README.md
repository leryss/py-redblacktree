Pure python3 implementation of a red black tree can be used as a set or a dictionary or both at the same time. It is tested and stable.
If you need any feature implemented or god forbid find a bug just open an issue on https://github.com/leryss/my-pypackages 

# Installation

    pip install redblacktree
    
# Examples
```py
>>> from redblacktree import rbtree

# Can initialize using lists, dicts or other iterables
>>> rbtree([1,2,3,4,5,6,7,8])
>>> Depth=4
           ┌───────────────4───────────────┐
   ┌───────2───────┐               ┌───────6───────┐
   1               3               5               7───┐
                                                       8
# You can store values with keys or not or both
>>> rbtree([1,2,(3,'three'),4,5,(6,'six'),7,8])
>>> Depth=4
           ┌───────────────4───────────────┐
   ┌───────2───────┐               ┌─────6:six─────┐
   1            3:three            5               7───┐
                                                       8
```

```py
# Inserting
tree.insert(5) # Inserts a key with no value
tree[5] = None # Same as above
tree.insert(5, 'five') # Inserts a key with value
tree[5] = 'five' # Same as above

# Removing
tree.remove(5)
del tree[5]

# Query
tree[5] # Returns value of key 5 from tree
5 in tree # Returns True if there is a key 5 in the tree

# Slicing also supported
tree[5:10] # Returns (key, value) pairs of all keys >= 5 and <= 10
tree[:5] # All (key, value) pairs for keys <= 5
# Beware slicing with a step for example tree[1:100:-1] wont work

# Simple iteration:
for k, v in tree:
    print(k, v)
```