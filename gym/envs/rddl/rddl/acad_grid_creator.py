import numpy as np

rows = 5
cols=5

for i in range(1, rows+1):
	for j in range(1, cols+1):
		# print("NOISE-PROB(x%d,y%d) = %8f;"%(i,j,np.random.uniform(low=0.04, high=0.065)))
		# print("NOISE-PROB(x%d,y%d) = 0.;"%(i,j))

		# if i>1:
			# print("PREREQ(CS%d%d,CS%d%d);"%(i,j,i-1,j))
		# 	# print("PREREQ(CS%d%d,CS%d%d);"%(i,j,i-1,j))
		# 	print("PREREQ(CS%d%d,CS%d%d);"%(i,j,i-1,j))
		# 	print("PREREQ(CS%d%d,CS%d%d);"%(i,j,i-1,j))
		# if j>1:
		# 	print("NEIGHBOR(x%d,y%d,x%d,y%d);"%(i,j,i,j-1))
		if i<rows-1:
			print("PREREQ(CS%d%d,CS%d%d);"%(i,j,i+1,j))
		if j<cols-1:
			print("PREREQ(CS%d%d,CS%d%d);"%(i,j,i,j+1))
