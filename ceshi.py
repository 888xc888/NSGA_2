class AA(object):
    def __init__(self, a, name):
        self.a = a
        self.name = name


A = AA(7, 'A')
d = AA(999, 'd')
t = AA(100, 't')
u = AA(-999, 'u')
k = AA(0, 'k')
b = AA(66, 'b')

value = [A, d, t, u, k, b]
print([x.name for x in value])

value.sort(key=lambda x: x.a, reverse=True)  # 使用distance属性值排序 # sort默认升序

print([x.name for x in value])
