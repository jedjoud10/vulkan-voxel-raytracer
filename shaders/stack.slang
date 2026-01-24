
public struct Stack<let N: int, T> {
    T[N] elements;
    uint8_t length;

    __init() {
        length = 0;
    }

    [mutating]
    void append(T element) {
        if (length == N) {
            printf("cannot append to stack. stack full. length=%i", length);
            return;
        }

        elements[length] = element;
        length++;
    }

    T peek() {
        return elements[length - 1];
    }

    [mutating]
    void pop() {
        if (length == 0) {
            return;
        }

        length--;
    }

    [mutating]
    bool tryDequeue(out T element) {
        if (length == 0) {
            return false;
        }

        element = elements[length-1];
        length--;
        return true;
    }
}



public struct SortedDistancesStack<let N: int, T> {
    T[N] elements;
    half[N] distances;
    uint length;

    __init() {
        length = 0;
    }

    [mutating]
    void swap(int a, int b) {
        half tmp_dist = distances[a];
        T tmp_packed = elements[a];

        distances[a] = distances[b];
        elements[a] = elements[b]; 
        
        distances[b] = tmp_dist;
        elements[b] = tmp_packed; 
    }

    [mutating]
    void append(T element, float tmin) {
        if (length == N) {
            //printf("cannot append to stack. stack full. length=%i", length);
            return;
        }

        distances[length] = tmin;
        elements[length] = element;

        // do swaps until it is in the right spot...
        int cur_index = length;
        while (cur_index >= 1 && distances[cur_index - 1] < distances[cur_index]) {
            swap(cur_index - 1, cur_index);
            cur_index = cur_index - 1;
        }

        length++;
    }

    [mutating]
    bool tryDequeue(out T element, out float tmin) {
        if (length == 0) {
            return false;
        }

        element = elements[length-1];
        tmin = distances[length-1];
        length--;
        return true;
    }
}