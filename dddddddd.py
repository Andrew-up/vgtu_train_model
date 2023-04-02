#

# version_1 = '.'.join('1.0000.1')
# # version_2 = ''.join('1.0.1'.split('.')).strip("0")
# print(version_1)
# print(version_1 == version_2)
# version_2 = '1.0.1.1'.replace('.', '')
# version_1 = int(version_1.strip("0"))
# version_2 = int(version_2.strip("0"))
# print(version_1 == version_2)


a = '1.0.1.0000.0.0.000'
b = '1.0.1.00001'
ind = 0
for i in range(-1, -len(a), -1):
    if a[i] in ('123456789'):
        ind = i
        break
print(a[: ind + 1])  # 1.0.1

ind2 = 0
for i in range(-1, -len(b), -1):
    if a[i] in ('123456789'):
        ind2 = i
        break
print(b[: ind2 + 1])
print(a == b)  # false

#
# version_1 = ''.join(reversed(version_1))
# # version_2 = '1.0.1'.replace('.', '')
#
#
# print(version_1.split('.'))


# a = '1.0.1.1110000.0.0.0'

# print(version_1[-1])
# print(version_2)
# print(version_1 <= version_2)
