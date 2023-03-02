from random import randint

numbers = [randint(0,100) for _ in range(25)]
print(numbers)
counter = 0
file_ = open("data.csv","w+")
for i in range(5):
     for j in range(5):
          file_.write(f"{numbers[counter]}")
          if j == 4:
               file_.write("\n")
          else:
               file_.write(",")
          counter += 1
file_.close()
