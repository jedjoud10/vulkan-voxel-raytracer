
struct Queue<let N: int, T> {
    T[N] elements;
    uint readHead;
    uint writeHead;
    uint length;

    __init() {
        length = 0;
        writeHead = 0;
        readHead = 0;
    }

    [mutating]
    void append(T element) {
        if (length == N) {
            // printf("queue full");
            return;
        }

        elements[writeHead] = element;
        writeHead++;
        length++;
        writeHead %= N;
    }

    [mutating]
    bool tryDequeue(out T element) {
        if (readHead == writeHead) {
            return false;
        }

        element = elements[readHead];
        readHead++; 
        readHead %= N; 
        length--;
        return true;
    }
}