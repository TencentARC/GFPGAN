from module_gfpgan import main
from threading import Thread
from io import StringIO
import os, sys, json, traceback
from tkinter import (
    Tk,
    Toplevel,
    PhotoImage,
    filedialog,
    messagebox,
    Button,
    Label,
    Frame,
    Entry,
    OptionMenu,
    Checkbutton,
    Spinbox,
    StringVar,
    BooleanVar,
    IntVar,
    DoubleVar,
    scrolledtext,
    )

# empty log.txt
with open("log.txt","w") as f:
    None
    
class Root(Tk):
    def __init__(self):
        super().__init__()

        # ----------   load setting variables -------------
        
        self.bg_upsampler = BooleanVar(value=False)
        self.version = StringVar()
        self.mystdout = ""
        self.old_stdout = ""
        try:
            with open("assets/settings.json", "r") as f:
                sett = json.load(f)
            self.tile = sett["tile"]
            self.upscale = sett["upscale"]
            self.suffix = sett["suffix"]
            self.extension = sett["extension"]
            self.weight = sett["weight"]
            self.selected_languaje = sett["language"]
            self.available_languages = sett["available_languages"]
            self.language(self.selected_languaje)# execute language funtion
            self.output = sett["output"]
        except:
            # default setting
            self.tile = 400
            self.upscale = 2
            self.suffix = "out"
            self.extension = "auto"
            self.weight = 0.5
            self.language("en")
            self.selected_languaje = "en"
            self.available_languages = ["en","es",]
            self.output = os.path.expanduser(os.path.join("~", "Documents", "Photo Restorer"))
        
            
        # load images
        self.bg_img = PhotoImage(file="assets/bg.png")
        self.wm_iconbitmap(bitmap="assets/icon.ico")

        
        # --------  create all buttons and frames in main menu  ----------
        
        # create main frame background
        self.main_frame = Frame(width = 406,height = 326)
        self.main_frame.pack_propagate(False)
        self.main_frame.pack()
        Label(self.main_frame, image=self.bg_img, bg="#00ffff", width = 406, height = 326).place(x=0, y=0)
        Frame(self.main_frame, bg="black", height=1).pack(pady=22)# separator
        # create loading message
        self.frame_loading = Frame(bg="#000022", width = 406, height = 326)
        Label(self.frame_loading, image=self.bg_img, width = 406,height = 326).place(x=0,y=0)
        self.loading = Label(self.frame_loading, bg="#000022", fg="light blue", font=("console",15))
        self.loading.place(x=110, y=60)
        # create console log
        self.log = scrolledtext.ScrolledText(
            self.frame_loading,
            bg="#000022",
            fg="light blue",
            width=42,
            height=9,
            highlightbackground="#00bbff",
            highlightcolor="#00bbff",
            selectbackground="gray",
            highlightthickness=1,
            )
        self.log.place(x=24, y=120)




        # button and bind effect
        b1 = Button(
            self.main_frame,
            text=self.lan["selectimages"],
            command=self.files,
            bg="#4488ee",
            activebackground="#66bbdd",
            fg="white",
            font=("Arial", 14),
            width=26,
            )
        b1.pack(anchor="center")
        def bin1(event):
            if event.type.value == str(7):
                b1["bg"] = "#44aaee"
            else:
                b1["bg"] = "#4488ee"
        b1.bind("<Enter>",bin1)
        b1.bind("<Leave>",bin1)
        
        #------------
        bar = Frame(self.main_frame)
        bar.place(x=2,y=2)
        Button(
            bar,
            text=self.lan["setting"],
            command=self.setting,
            bg="black",
            fg="white",
            activebackground="orange",
            font=("terminal", 9)
            ).grid(row=0, column=0)
        Button(
            bar,
            text=self.lan["about"],
            command=self.about,
            bg="black",
            fg="white",
            activebackground="orange",
            font=("terminal", 9)
            ).grid(row=0, column=1)
        Button(
            bar,
            text=self.lan["help"],
            command=self.help,
            bg="black",
            fg="white",
            activebackground="orange",
            font=("terminal", 9)
            ).grid(row=0, column=2)
        
        
        # box1
        box1 = Frame(self.main_frame, bg="#001122", bd=8, relief="ridge", width=300, height=100)
        box1.pack()
        box1.pack_propagate(False)
        Label(box1, text=self.lan["facerestorer"], bg="#001122", fg="white", font=("Arial", 14)).pack(f="both")
        # box2
        box2 = Frame(self.main_frame, bg="#001122", bd=8, relief="ridge", width=300, height=100)
        box2.pack()
        box2.pack_propagate(False)
        Label(box2, text=self.lan["bgrestorer"], bg="#001122", fg="white", font=("Arial", 14)).pack(f="both")

        # ---------------------- check if models exist --------------------------
        models = []
        if os.path.exists("experiments/pretrained_models/GFPGANv1.4.pth"):
            models.append("1.4")
        if os.path.exists("experiments/pretrained_models/GFPGANv1.3.pth"):
            models.append("1.3")
        if os.path.exists("experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth"):
            models.append("1.2")
        if os.path.exists("experiments/pretrained_models/RestoreFormer.pth"):
            models.append("RestoreFormer")
        try:
            self.version.set(models[0])
        except:
            models.append("1.4")
            self.version.set("1.4")
            messagebox.showerror('Error','No models availables\n please download models to :\n /experiments/pretrained_models/')
        # menu model
        Label(box1, text=self.lan["models"], bg="#001122", fg="#88eeff").pack()
        model = OptionMenu(box1, self.version,*models)
        model.pack()
        model.configure(
        bg="gray",
        fg="yellow",
        width=15,
        activebackground="orange",
        highlightbackground="green",
        highlightthickness=3,
        )
        
        
        # ------ checkbox bg_upsampler 
        ch = Checkbutton(
            box2,
            text=self.lan["bgupsampler"],
            variable=self.bg_upsampler,
            selectcolor="orange",
            indicatoron=True,
            background="#8899aa",
            activebackground="#77aaee",
            )
        ch.pack()
        def fun(e):
            if self.bg_upsampler.get():
                ch["bg"] = "#8899aa"
            else:
                ch["bg"] = "#00ffff"
        ch.bind("<Button-1>",fun)



    # ------------  Create all funtions  --------------
    
    def stop(self):
        self.stop = True
        self.destroy()

    def language(self, name):
        # load built in language english
        if name == "en":
            self.lan = {
                "setting":"Setting",
                "about":"About",
                "help":"Help",
                "selectimages":"Select Images",
                "facerestorer":"Face Restorer",
                "models":"Models Availables",
                "bgrestorer":"Background Restorer",
                "bgupsampler":"Background Upsampler",
                "upscale":"Upscale",
                "suffix":"Suffix",
                "extension":"Output Image Extension",
                "weight":"Adjustable Weight",
                "bgtile":"Background Tile",
                "ok":"Ok",
                "reset":"Reset",
                "cancel":"Cancel",
                "programming":"<>Programming<>",
                "userinterface":"User Interface",
                "funtionandmodel":"Funtion and Models",
                "language":"language",
                "output":"Output",
                "processing":"Processing",
                "of":" Of ",
                "help_text":("""
    GFP-GAN: Face Restoration with Artificial Intelligence
    Real-ESRGAN background Restoration with Artificial Intelligence
    this last funtion is slower in CPU, is recommended to use GPU.
    """
                             )
                }
            
        elif name == "es":
            # load built in language spanish
            self.lan = {
                "setting":"Configuracion",
                "about":"Acerca",
                "help":"Ayuda",
                "selectimages":"Selecionar Imagenes",
                "facerestorer":"Restauracion Facial",
                "models":"Modelos del Restaurador",
                "bgrestorer":"Restauracion de Fondo",
                "bgupsampler":"Restaurar fondo",
                "upscale":"Escalar",
                "suffix":"Sufijo de nombre",
                "extension":"Extension de Salida",
                "weight":"Peso Ajustable",
                "bgtile":"Azulejo de Fondo",
                "ok":"Guardar",
                "reset":"Resetear",
                "cancel":"Cancelar",
                "programming":"<>Programacion<>",
                "userinterface":"Interfaz de Usuario",
                "funtionandmodel":"Funciones y Modelos",
                "language":"lenguaje",
                "output":"Salida",
                "processing":"Procesando",
                "of":" De ",
                "help_text":("""
    GFP-GAN: Restauracion Facial con tecnologia de inteligencia artificial
    
    Real-ESRGAN: Restauracion de Fondo con tecnologia de inteligencia artificial
    esta ultima funcion es mas lenta en CPU, se recomienda usar GPU.
    """
                             )
                }
            
        # try to load lenguage from file replace the previus loaded language
        try:
            with open(f"assets/{name}.json", "r") as f:
                l = json.load(f)
                # load only if keys are correct
                if l.keys() == self.lan.keys():
                    self.lan = l
                else:
                    messagebox.showerror("Error",f"File assets/{name}.json dictionary keys are incorrect")
        except:
            # if file not exist create new file so user can edit and customize.
            with open(f"assets/{name}.json", "w") as f:
                s = json.dumps(self.lan)
                f.write(s)



 
    def process(self, input='inputs/whole_imgs'):

        if self.bg_upsampler.get():
            upsampler = "realesrgan"
        else:
            upsampler = "no"

        suff = self.suffix
        if not suff:
            suff = None

        main(
            input = input,
            output = self.output,
            version = self.version.get(),
            upscale = self.upscale,
            bg_upsampler = upsampler,# default realesrgan
            bg_tile = self.tile,# 400
            suffix = suff,
            ext = self.extension,
            weight = self.weight,#0.5
            )

    # pick up images
    def files(self):
        self.stop = False
        lista = filedialog.askopenfiles(
            filetypes=(
                "Images *.jpg",
                "Images *.jpeg",
                "Images *.png",
                "Images *.bmp",
                "All *.*",
                )
            )
        if not lista:
            return
        
        def pro(*args):
            self.main_frame.pack_forget()
            self.frame_loading.place(x=0,y=0)
            # redirect console output to variable to use in tkinter console
            self.old_stdout = sys.stdout
            sys.stdout = self.mystdout = StringIO()
            n=1
            for foto in args:
                if self.stop:
                    break
                self.loading["text"] = self.lan["processing"] + " " + str(n) + self.lan["of"] + str(len(lista))
                
                
                # start restore process
                try:
                    self.process(input=foto.name)
                except:
                    #  show error message in console and dialog, save image process log to file.
                    messagebox.showerror("Error", f"Corrupt Image\n {foto.name}")
                    with open("log.txt","a") as f:
                        f.write("\n\nError Can Not Process : " + foto.name + "\n")
                        traceback.print_exc(file=f)
                    print("Error can't process ",foto.name)
                n+=1
                
            # restore output messages to python console
            sys.stdout = self.old_stdout
            self.frame_loading.place_forget()
            self.main_frame.pack(fill="both")
            self.stop = True
            
        p = Thread(target=pro, args=(lista))
        p.start()
        
        # insert text to the log console
        def info():
            try:
                self.log.delete(0.0,"end")
                self.log.insert("end", self.mystdout.getvalue())
                self.log.see("end")
            except:
                None
            if self.stop:
                return
            self.timer = self.after(3000, info)
        info()



        
    def about(self):
        bg_color = "#666666"
        dialog = Toplevel()
        dialog.geometry(f"220x310+{self.winfo_x()+100}+{self.winfo_y()+40}")
        dialog.overrideredirect(True)
        dialog.configure(bg=bg_color, relief="ridge", bd=4)
        
        Label(dialog, text="GFPGAN v1.3.8", bg=bg_color, fg="light blue", font=("Arial", 16)).pack(pady=10)
        Label(dialog, text=self.lan["programming"], bg=bg_color, fg="#00ff00", font=("Arial", 12)).pack()
        Label(dialog, text=self.lan["userinterface"], bg=bg_color, fg="#66ddff", font=("Arial", 12)).pack()
        Label(dialog, text="Erick Esau Martinez\n", bg=bg_color, fg="pink", font=("Arial", 12)).pack()
        Label(dialog, text=self.lan["funtionandmodel"], bg=bg_color, fg="#66ddff", font=("Arial", 12)).pack()
        Label(dialog, text="Xintao Wang \n Yu Li \n Honglun Zhang \n Ying Shan", bg=bg_color, fg="#ffdddd", font=("Arial", 12)).pack()
        
        Button(dialog, text="X", bg="red", activebackground="orange", command=dialog.destroy).place(x=190, y=0)
        dialog.grab_set()


    def help(self):
        bg_color = "#666666"
        dialog = Toplevel()
        dialog.geometry(f"220x310+{self.winfo_x()+100}+{self.winfo_y()+40}")
        dialog.overrideredirect(True)
        dialog.configure(bg=bg_color, relief="ridge", bd=4)
        Label(dialog, text = self.lan["help"], bg=bg_color, fg="#66aaff", font = ("console", 16)).pack(pady=10)
        Label(dialog, text = self.lan["help_text"], bg=bg_color, fg="white", wraplen=220).pack()
        Button(dialog, text="X", bg="red", activebackground="orange", command = dialog.destroy).place(x=190, y=0)
        dialog.grab_set()
        

    def setting(self):
        bg = "#666666"
        self.dialogo = Toplevel()
        self.dialogo.geometry(f"220x358+{self.winfo_x()+100}+{self.winfo_y()+40}")
        self.dialogo.overrideredirect(True)
        self.dialogo.configure(bg=bg, relief="ridge", bd=4)
        # frames
        box1 = Frame(self.dialogo, bd=5, relief="ridge", bg="gray")
        box1.pack()
        box2 = Frame(self.dialogo, bd=5, relief="ridge", bg="gray")
        box2.pack(pady=8)
        
        # create setting vars
        upscale = IntVar(value=self.upscale)
        extension = StringVar(value=self.extension)
        weight = DoubleVar(value=self.weight)
        tile = IntVar(value=self.tile)
        lan = StringVar(value=self.selected_languaje)

        # ------ upscale spinbox
        Label(box1, text=self.lan["upscale"], bg=bg,fg="white", width=25).grid(sticky="we")
        Spinbox(box1, from_=0, to=100, textvariable=upscale, bg="light blue", buttonbackground="yellow").grid(sticky="we")

        # ------ suffix
        Label(box1, text=self.lan["suffix"], bg=bg,fg="white").grid(sticky="we")
        suffix = Entry(box1, bg="light blue")
        suffix.insert(0,self.suffix)
        suffix.grid(sticky="we")

        # ------ Image extension
        Label(box1, text=self.lan["extension"], bg=bg,fg="white").grid(sticky="we")
        extension_list = ['auto','jpg', 'png']

        extension_menu = OptionMenu(box1, extension, *extension_list)
        extension_menu.grid(sticky="we")
        extension_menu.configure(
        bg=bg,
        fg="yellow",
        activebackground="orange",
        highlightbackground="green",
        highlightthickness=3,
        )

        # output path
        box3 = Frame(box1, bg="black", relief="ridge", height=25)
        box3.grid_propagate(False)
        box3.grid(sticky="we", pady=5)
        def out():
            f = filedialog.askdirectory()
            if f:
                output.delete(0,last="end")
                output.insert(0,f)
                
        output = Entry(box3, font=("console",10),width=15)
        output.grid(column=0, row=0)
        output.insert(0, self.output)
        Button(box3, text=self.lan["output"], bg="light blue",width=10,font=("console",8), command=out).grid(column=1, row=0)

        # ------ weight
        Label(box1, text=self.lan["weight"], bg=bg,fg="white").grid(sticky="we")
        weight_spin = Spinbox(
            box1,
            values=(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1),
            textvariable=weight,
            bg="light blue",
            buttonbackground="yellow",
            )
        weight_spin.grid(sticky="we")
        weight.set(self.weight)

        # ------ bg tile
        Label(box2, text=self.lan["bgtile"], bg=bg,fg="white", width=25).grid(sticky="we")
        Spinbox(
            box2,
            from_=0,
            to=5000,
            textvariable=tile,
            bg="light blue",
            buttonbackground="yellow",
            ).grid(sticky="we")
        
        # menu de lenguajes
        box4 = Frame(self.dialogo, bg=bg, relief="ridge", bd=1)
        box4.pack()
        Label(box4, text=self.lan["language"], bg=bg,fg="white", width=15).grid(row=0, column=0)
        lan_menu = OptionMenu(box4, lan, *self.available_languages)
        lan_menu.grid(row=0, column=1)
        lan_menu.configure(
        bg="gray",
        fg="yellow",
        width=5,
        activebackground="orange",
        highlightbackground="green",
        highlightthickness=3,
        )


        def save():
            self.upscale = upscale.get()
            self.suffix = suffix.get()
            self.extension = extension.get()
            self.weight = weight.get()
            self.tile = tile.get()
            self.selected_languaje = lan.get()
            self.output = output.get()
            sett = {
                "upscale":self.upscale,
                "suffix":self.suffix,
                "extension":self.extension,
                "weight":self.weight,
                "tile":self.tile,
                "language":self.selected_languaje,
                "available_languages":self.available_languages,
                "output":self.output,
                }
            sett = json.dumps(sett)
            with open("assets/settings.json", "w") as f:
                f.write(sett)
            self.dialogo.destroy()

        def reset():
            upscale.set(2)
            extension.set("auto")
            weight.set(0.5)
            tile.set(400)
            suffix.delete(0,last="end")
            suffix.insert(0,"out")
            output.delete(0,last="end")
            output.insert(0, os.path.expanduser(os.path.join("~", "Documents", "Photo Restorer")))

        Button(
            self.dialogo,
            text=self.lan["ok"],
            command=save,
            bg="#4477dd",
            fg="white",
            relief="raised",
            borderwidth=3,
            activebackground="#66aaff",
            width=8,
            ).place(x=8,y=320)
        Button(
            self.dialogo,
            text=self.lan["reset"],
            command=reset,
            bg="#4477dd",
            fg="white",
            relief="raised",
            borderwidth=3,
            activebackground="#66aaff",
            width=8,
            ).place(x=74,y=320)
        Button(
            self.dialogo,
            text=self.lan["cancel"],
            command=lambda x=None:(self.dialogo.destroy(),),
            bg="#4477dd",
            fg="white",
            relief="raised",
            borderwidth=3,
            activebackground="#66aaff",
            width=8,
            ).place(x=136,y=320)

        
        self.dialogo.grab_set()




try:
    window = Root()
    window.protocol("WM_DELETE_WINDOW", window.stop)
    window.title("Photo Restorer GFPGAN Artificial Intelligence")
    window.configure(bg="gray")
    window.geometry("410x330+400+322")
    window.resizable(0,0)
    if __name__ == "__main__":
        window.mainloop()
except:
    with open("log.txt","a") as f:
        f.write("\n")
        traceback.print_exc(file=f)
        raise
