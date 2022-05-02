from configs.config import CFG

from model.neural_net import Dog


print (CFG)


buddy = Dog(CFG,"Buddy", 9)
miles = Dog(CFG,"Miles", 2)

print (buddy.name)
print (buddy.species)
print(buddy.batch_size)

#print(buddy)