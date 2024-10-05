def issorted(numbers):

     size_of_list = len(numbers)

     for i in range(1, size_of_list):
          if numbers[i-1] > numbers[i]:
               return False
     return True
