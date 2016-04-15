with open("chars74k-lite/a/a_0.jpg", "rb") as imageFile:
  f = imageFile.read()
  b = bytearray(f)

line = []
print(len(b))
for i in range(len(b)):
    line.append(b[i])
    if i%20 == 19:
        print(line)
        line = []

# Test