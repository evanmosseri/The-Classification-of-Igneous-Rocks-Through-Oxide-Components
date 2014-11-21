from Tkinter import *
from ttk import *
import tkFileDialog

class App:
	def get_filename(self):
		self.button2.config(text=tkFileDialog.askopenfilename())
	def __init__(self, master):
		self.master = master

		master.geometry("640x480")

		self.frame = Frame(master)
		self.frame.pack()

		self.button = Button(self.frame, text="QUIT", command=self.frame.quit)
		self.button.pack()

		self.button2 = Button(self.frame, text="Get Filename", command=self.get_filename)
		self.button2.pack()


		menubar = Menu(root)

		filemenu = Menu(menubar, tearoff=0)
		filemenu.add_command(label="Open", command=lambda x: x)
		filemenu.add_command(label="Save", command=lambda x: x)
		filemenu.add_separator()
		filemenu.add_command(label="Exit", command=root.quit)
		menubar.add_cascade(label="File", menu=filemenu)

		# create more pulldown menus
		editmenu = Menu(menubar, tearoff=0)
		editmenu.add_command(label="Cut", command=lambda x: x)
		editmenu.add_command(label="Copy", command=lambda x: x)
		editmenu.add_command(label="Paste", command=lambda x: x)
		menubar.add_cascade(label="Edit", menu=editmenu)

		helpmenu = Menu(menubar, tearoff=0)
		helpmenu.add_command(label="About", command=lambda x: x)
		menubar.add_cascade(label="Help", menu=helpmenu)



		root.config(menu=menubar)


		master.mainloop()
		master.destroy()

root = Tk()

app = App(root)


