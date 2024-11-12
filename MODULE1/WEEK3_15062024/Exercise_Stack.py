class MyStack:
    def __init__(self, capacity):
        self.capacity = capacity
        self.stack_list = []

    def is_empty(self):
        if len(self.stack_list) == 0:
            return True
        else:
            return False

    def is_full(self):
        if len(self.stack_list) == self.capacity:
            return True
        else:
            return False

    def pop(self):
        top_value = self.stack_list[-1]
        self.stack_list.pop(-1)
        return top_value

    def push(self, value):
        self.stack_list.append(value)

    def top(self):
        return self.stack_list[-1]


stack1 = MyStack(capacity=5)
stack1.push(2)
stack1.push(4)
print(stack1.top())
stack1.push(5)
print(stack1.top())
stack1.pop()
print(stack1.top())
print(stack1.is_full())
print(stack1.is_empty())
