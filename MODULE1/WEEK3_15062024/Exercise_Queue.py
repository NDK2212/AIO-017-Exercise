class MyQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue_list = []

    def is_empty(self):
        return len(self.queue_list) == 0

    def is_full(self):
        return len(self.queue_list) == self.capacity

    def dequeue(self):
        self.queue_list.pop(-1)

    def enqueue(self, value):
        self.queue_list.insert(0, value)

    def front(self):
        return self.queue_list[-1]


queue1 = MyQueue(capacity=5)
queue1.enqueue(2)
queue1.enqueue(4)
print(queue1.front())
queue1.enqueue(5)
print(queue1.front())
queue1.dequeue()
print(queue1.front())
print(queue1.is_full())
print(queue1.is_empty())
