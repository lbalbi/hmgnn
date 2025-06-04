class Logger:

    def __init__(self, name):
        self.name = name
        self.log_file = f"{name}.log"
        self.file = open(self.log_file, 'w')
        self.file.write(f"Results for {name}\n")
        self.file.write("=" * 50 + "\n")

    def log(self, message):
        print(message)
        self.file.write(message + "\n")
        self.file.flush()

    def close(self):
        self.file.close()