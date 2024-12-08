import pickle
import tkinter as tk
import imageio
from skimage.transform import rescale
import numpy as np
from PIL import Image, ImageTk

tkinter_color_list = ['red', 'green', 'blue', 'orange', 'pink', 'dark green', 'dark grey', 'gold', 'gray', 'light blue', 'cyan', 'light green', 'magenta', 'purple',  'white', 'yellow']
tkinter_current_color = 0

class Rectangle:
    def __init__(self, canvas_id=None, first_corner=(0, 0), second_corner=(0, 0), name="", color=None):
        global tkinter_current_color, tkinter_color_list
        self.name = name
        self.canvas_id = canvas_id
        self.first_corner = first_corner
        self.second_corner = second_corner
        if color is None:
            self.color = tkinter_color_list[tkinter_current_color]
            tkinter_current_color = (tkinter_current_color + 1) % len(tkinter_color_list)
        else:
            self.color = color

    def __str__(self):
        return self.name + " " + f'({self.first_corner[0]:1.2f},{self.first_corner[1]:1.2f}),({self.second_corner[0]:1.2f},{self.second_corner[1]:1.2f}) ' + self.color

    def area(self, width, height):
        return abs(self.second_corner[0] - self.first_corner[0]) * width \
               * abs(self.second_corner[1] - self.first_corner[1]) * height

class RectangleDrawer:
    def __init__(self, image_file, root=None, rectangles=None):
        # Store a reference to the root window and the image
        flag = False
        if root is None:
            flag = True
            root = tk.Tk()
            root.title("My App")

        self.canvas_size = (800, 600)
        self.image_file = image_file
        self.root = root
        self.image_raw = imageio.imread(image_file)
        self.image_resized = self.resize_image()
        self.image = ImageTk.PhotoImage(Image.fromarray(self.image_resized))

        # Create a canvas to display the image

        self.im_width = self.image.width()
        self.im_height = self.image.height()
        self.canvas = tk.Canvas(root, width=self.im_width, height=self.im_height)
        self.canvas.pack(side='left')
        self.canvas.create_image(0, 0, anchor='nw', image=self.image)

        # Bind mouse events to the canvas
        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_button_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Create a frame to hold the list of rectangles and the controls
        self.frame = tk.Frame(root)
        self.frame.pack(side='right', fill='y')

        # Create a listbox to display the rectangles
        self.listbox = tk.Listbox(self.frame, selectmode='single', width=50)
        self.listbox.pack(fill='both')

        # Create a LabelFrame to display selected rectangle properties
        self.label_frame = tk.LabelFrame(self.frame, text='Rectangle Properties')
        self.label_frame.pack(fill='both')
        # add label and entry for rectangle name
        name_label = tk.Label(self.label_frame, text='Name')
        name_label.grid(row=0, column=0)
        self.name_entry = tk.Entry(self.label_frame)
        self.name_entry.grid(row=0, column=1)
        self.listbox.bind('<<ListboxSelect>>', self.onselect)
        self.name_entry.bind('<Return>', self.on_name_change)


        # Create buttons for common actions
        #tk.Button(self.frame, text="Add", command=self.add_rectangle).pack()
        self.delete_button = tk.Button(self.label_frame, text="Delete", command=self.delete_rectangle)
        self.delete_button.enable = False
        self.delete_button.grid(row=1, column=0)

        self.set_frame_state(self.label_frame, 'disabled')
        #tk.Button(self.frame, text="Rename", command=self.rename_rectangle).pack()

        self.save_button = tk.Button(self.frame, text="Save", command=self.save)
        self.save_button.pack()

        # Initialize the list of rectangles
        self.rectangles = []

        if rectangles is not None:
            for rect_name in rectangles:
                r = Rectangle(name=rect_name, first_corner=rectangles[rect_name][0], second_corner=rectangles[rect_name][1])
                r.canvas_id = self.canvas.create_rectangle(
                    r.first_corner[0] * self.im_width,
                    r.first_corner[1] * self.im_height,
                    r.second_corner[0] * self.im_width,
                    r.second_corner[1] * self.im_height,
                    outline=r.color
                )
                self.rectangles.append(r)
                self.listbox.insert(tk.END, r)

        # Initialize the state for drawing a rectangle
        self.drawing_rectangle = False
        self.current_rectangle = None

        if flag:
            root.mainloop()

    def set_frame_state(self, frame, state):
        for child in frame.winfo_children():
            child.configure(state=state)

    def set_selected_rectangle(self, index):
        rect = self.rectangles[index]
        self.current_rectangle = rect
        # set name_entry value equals to the name of the selected rectangle
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, rect.name)
        self.set_frame_state(self.label_frame, 'normal')

    def onselect(self, event):
        w = event.widget
        index = int(w.curselection()[0])
        self.set_selected_rectangle(index)

    def delete_rectangle(self):
        if self.current_rectangle is not None:
            self.canvas.delete(self.current_rectangle.canvas_id)
            self.listbox.delete(tk.ACTIVE)
            self.rectangles.remove(self.current_rectangle)
            self.current_rectangle = None
            self.set_frame_state(self.label_frame, 'disabled')

    def resize_image(self):
        imw = self.image_raw.shape[1]
        imh = self.image_raw.shape[0]
        rw = imw / self.canvas_size[1]
        rh = imh / self.canvas_size[0]
        max_r = max(rh, rw)
        if max_r > 1:
            return (rescale(self.image_raw, 1/max_r, channel_axis=2)*255).astype(np.uint8)
        else:
            return self.image_raw

    def on_button_press(self, event):
        # Start drawing a rectangle
        self.drawing_rectangle = True
        self.current_rectangle = Rectangle(first_corner=(event.x/self.im_width, event.y/self.im_height),
                                           second_corner=(event.x/self.im_width, event.y/self.im_height))

    def on_name_change(self, event):
        self.current_rectangle.name = self.name_entry.get()
        self.listbox.delete(tk.ACTIVE)
        self.listbox.insert(tk.ACTIVE, self.current_rectangle)

    def on_button_move(self, event):
        # Update the rectangle as the mouse is moved
        if self.drawing_rectangle:
            # Delete the previous rectangle, if any
            if self.current_rectangle.canvas_id is not None:
                self.canvas.delete(self.current_rectangle.canvas_id)

            self.current_rectangle.second_corner = (event.x/self.im_width, event.y/self.im_height)
            # Create a new rectangle
            self.current_rectangle.canvas_id = self.canvas.create_rectangle(
                self.current_rectangle.first_corner[0]*self.im_width,
                self.current_rectangle.first_corner[1]*self.im_height,
                self.current_rectangle.second_corner[0] * self.im_width,
                self.current_rectangle.second_corner[1] * self.im_height,
                outline=self.current_rectangle.color
            )

    def on_button_release(self, event):
        # Finish drawing the rectangle
        self.drawing_rectangle = False

        if self.current_rectangle.area(self.im_width, self.im_height) > 10:
            self.rectangles.append(self.current_rectangle)
            self.listbox.insert(tk.END, self.current_rectangle)
            self.listbox.select_set(tk.END)
            self.set_selected_rectangle(len(self.rectangles)-1)
        else:
            self.canvas.delete(self.current_rectangle.canvas_id)
            self.current_rectangle = None

    def save(self):
        from tkinter import filedialog
        data = {}
        data["image"] = self.image_file

        rectangles = {}
        data["rectangles"] = rectangles
        #tkinter open savefile dialog
        filename = filedialog.asksaveasfilename(initialdir = ".",title = "Select file",filetypes = (("pickle files","*.pickle"),("all files","*.*")))
        for rect in self.rectangles:
            rectangles[rect.name] = (rect.first_corner, rect.second_corner)
        #pickle data
        with open(filename, 'wb') as f:
            pickle.dump(data, f)



if __name__ == "__main__":
    #image_file = r"screenshots\alliance_territory_cell.png"
    #image_file = r"screenshots\alliance_members_frame.png"
    #image_file = r"screenshots\city_info_frame2.png"


    #image_file = r"screenshots/alliance_members_frame_cell2.png"
    #image_file = r"screenshots\city_info_merite_frame.png"
    #image_file = r"screenshots\city_info_frame.png"

    #image_file = r"triangle_stats_11_2023/IMG_1799.png"
    #r = RectangleDrawer(image_file)
    #exit(0)

    #pfile = r"alliance_members_frame_cell.pickle"
    #pfile = r"alliance_members_frame.pickle"

    #image_file = r"triangle_stats_11_2023/IMG_1799.png"
    #pfile = r"triangle_stats.pickle"

    image_file = r"screenshots\annoucement_banner_new.png"
    pfile = r"city_info_frame.pickle"

    #image_file = r"screenshots\city_info_frame2.png"
    #pfile = r"city_info_frame2.pickle"

    #image_file = r"screenshots\merit_ranking_cell.png"
    #pfile = r"merit_ranking_cell.pickle"

    #image_file = r"screenshots\commander_ranking_city_cell.png"
    #pfile = r"ranking_cell.pickle"

    #image_file = r"screenshots\alliance_territory_cell.png"
    #pfile = r"alliance_territory_ranking_cell.pickle"

    #image_file = r"screenshots\alliance_elite_cell.png"
    #pfile = r"alliance_elite_ranking_cell.pickle"

    #image_file = r"screenshots\alliance_power_cell.png"
    #pfile = r"alliance_ranking_cell.pickle"

    #image_file = r"screenshots\commander_ranking_island_cell.png"
    #pfile = r"island_ranking_cell.pickle"

    #image_file = r"screenshots\cross_nation_commander_ranking_cell.png"
    #pfile = r"cross_nation_commander_ranking_cell.pickle"

    #image_file = r"screenshots\cross_nation_alliance_ranking_cell2.png"
    #pfile = r"cross_nation_alliance_ranking_cell.pickle"


    #image_file = r"screenshots\StrongestCommanderRanking2.png"
    #pfile = r"keranking_frame.pickle"

    #image_file = r"screenshots\ranking_merit_frame.png"
    #pfile = r"ranking_frame.pickle"


    ######################### Navigator frames
    #image_file = r"patterns/Navigator/frame_main_city.png"
    #pfile = r"patterns/Navigator/frame_main_city.pickle"

    #image_file = r"patterns/Navigator/frame_exit.png"
    #pfile = r"patterns/Navigator/frame_exit.pickle"

    #image_file = r"patterns/Navigator/frame_my_info.png"
    #pfile = r"patterns/Navigator/frame_my_info.pickle"

    #image_file = r"patterns/Navigator/frame_ranking_globe.png"
    #pfile = r"patterns/Navigator/frame_ranking_globe.pickle"

    #image_file = r"patterns/Navigator/frame_ranking_nation.png"
    #pfile = r"patterns/Navigator/frame_ranking_nation.pickle"

    #image_file = r"patterns/Navigator/frame_event_menu.png"
    #pfile = r"patterns/Navigator/frame_event_menu.pickle"

    import os
    new = not os.path.exists(pfile)
    if new:
        r = RectangleDrawer(image_file)
    else:

        with open(pfile, 'rb') as f:
            data = pickle.load(f)
        r = RectangleDrawer(image_file, rectangles=data["rectangles"])