from time import ctime



class ALFLogger():


	def __init__(self, output_path, jobID="XXX"):

		self.jobID = jobID
		self.name = f"ALFLogger_{self.jobID}"
		self.file = f"{output_path}/{self.name}.txt"

		self.w = [f"Init {self.name} at {ctime()}"]

	def saveLog(self):

		with open(self.file, "w") as f:
			f.write(f"Save at {ctime()}\n" + "\n".join(self.w))

	def log(self, t):
		self.w.append(t)
		self.saveLog()

