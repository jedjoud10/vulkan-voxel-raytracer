#ifndef OCTREE
#define OCTREE

public static const int3[] CHILDREN_OFFSETS = {
    int3(0, 0, 0),
    int3(1, 0, 0),
    int3(0, 0, 1),
    int3(1, 0, 1),
    int3(0, 1, 0),
    int3(1, 1, 0),
    int3(0, 1, 1),
    int3(1, 1, 1),
};

public struct Node {
	public int selfIndex;
	public int parentIndex;
	public int childBaseIndex; // set to -1 for leaf nodes
	public int padding_;
}

#endif