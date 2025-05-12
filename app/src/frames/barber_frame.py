import glob
import os
import random
from tkinter import filedialog
from types import SimpleNamespace
from typing import Optional, Callable

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image
from tkinterdnd2 import DND_FILES

from app.src.utils import resource_path


class BarberFrame(ctk.CTkToplevel):
    def __init__(self, master = None):
        super().__init__(master=master)

        self.orig_filepath: str = ""
        self._sync_method: Optional[Callable[[Image.Image, str], None]] = None

        self.dnd_light_img = Image.open(resource_path("dnd-light.png"))
        self.dnd_dark_img = Image.open(resource_path("dnd-dark.png"))

        self.geometry("1920x1000")

        self.orig_label = ctk.CTkButton(
            master=self,
            text="",
            text_color="#333333",
            height=450,
            width=600,
            border_width=1,
            image=ctk.CTkImage(
                light_image=self.dnd_light_img,
                dark_image=self.dnd_dark_img,
                size=(224, 224),
            ),
            hover=False,
            fg_color="transparent",
        )

        self.res_label = ctk.CTkButton(
            master=self,
            text="",
            text_color="#333333",
            height=450,
            width=600,
            border_width=1,
            image=ctk.CTkImage(
                light_image=self.dnd_light_img,
                dark_image=self.dnd_dark_img,
                size=(224, 224),
            ),
            hover=False,
            fg_color="transparent",
        )

        self.thresh_label = ctk.CTkButton(
            master=self,
            text="",
            text_color="#333333",
            height=450,
            width=600,
            border_width=1,
            image=ctk.CTkImage(
                light_image=self.dnd_light_img,
                dark_image=self.dnd_dark_img,
                size=(224, 224),
            ),
            hover=False,
            fg_color="transparent",
        )

        self.cc_label = ctk.CTkButton(
            master=self,
            text="",
            text_color="#333333",
            height=450,
            width=600,
            border_width=1,
            image=ctk.CTkImage(
                light_image=self.dnd_light_img,
                dark_image=self.dnd_dark_img,
                size=(224, 224),
            ),
            hover=False,
            fg_color="transparent",
        )

        self.cc_dil_label = ctk.CTkButton(
            master=self,
            text="",
            text_color="#333333",
            height=450,
            width=600,
            border_width=1,
            image=ctk.CTkImage(
                light_image=self.dnd_light_img,
                dark_image=self.dnd_dark_img,
                size=(224, 224),
            ),
            hover=False,
            fg_color="transparent",
        )

        # ========== BEGIN Block size parameter ==========
        min_block_size: int = 3
        max_block_size: int = 101
        block_size_step: int = 2
        default_block_size: int = 5

        self.thresh_block_size_var = ctk.IntVar(value=default_block_size)
        self.last_thresh_block_size_var = ctk.IntVar(value=default_block_size)

        self.thresh_block_size_name = ctk.CTkLabel(
            master=self,
            text="Block Size",
            height=25,
            width=75,
        )

        self.thresh_block_size = ctk.CTkSlider(
            master=self,
            from_=min_block_size,
            to=max_block_size,
            number_of_steps=(max_block_size-min_block_size) // block_size_step,
            variable=self.thresh_block_size_var,
            command=lambda x: self._update_thresh_block_size(x),
        )

        self.thresh_block_size_label = ctk.CTkLabel(
            master=self,
            text=str(self.thresh_block_size_var.get()),
        )

        self.thresh_block_size_down = ctk.CTkButton(
            master=self,
            text="<",
            width=25,
            height=25,
            command=lambda: self._update_thresh_block_size_via_button(-block_size_step),
        )

        self.thresh_block_size_up = ctk.CTkButton(
            master=self,
            text=">",
            width=25,
            height=25,
            command=lambda: self._update_thresh_block_size_via_button(block_size_step),
        )

        self.thresh_block_size_name.place(x=1275, y=45)
        self.thresh_block_size.place(x=1385, y=50)
        self.thresh_block_size_label.place(x=1585, y=45)
        self.thresh_block_size_down.place(x=1620, y=45)
        self.thresh_block_size_up.place(x=1650, y=45)
        # ========== END Block size parameter ==========

        # ========== BEGIN C parameter ==========
        min_c: int = 1
        max_c: int = 20
        c_step: int = 1
        default_c: int = 5

        self.thresh_c_var = ctk.IntVar(value=default_c)
        self.last_thresh_c_var = ctk.IntVar(value=default_c)

        self.thresh_c_name = ctk.CTkLabel(
            master=self,
            text="Constant",
            height=25,
            width=67,
        )

        def _update_c_size(position: int):
            if self.last_thresh_c_var.get() != position:
                self.last_thresh_c_var.set(int(position))
                self.thresh_c_label.configure(text="%.0f" % position)
                self._update_state()

        self.thresh_c = ctk.CTkSlider(
            master=self,
            from_=min_c,
            to=max_c,
            number_of_steps=(max_c - min_c) // c_step,
            variable=self.thresh_c_var,
            command=lambda x: _update_c_size(x),
        )

        self.thresh_c_label = ctk.CTkLabel(
            master=self,
            text=str(self.thresh_c_var.get()),
        )

        def _update_thresh_c_via_button(change: int):
            new_value = self.thresh_c_var.get() + change

            if self.thresh_c.cget('from_') <= new_value <= self.thresh_c.cget('to'):
                self.thresh_c_var.set(new_value)
                self.thresh_c_label.configure(text=new_value)
                self._update_state()

        self.thresh_c_down = ctk.CTkButton(
            master=self,
            text="<",
            width=25,
            height=25,
            command=lambda: _update_thresh_c_via_button(-c_step),
        )

        self.thresh_c_up = ctk.CTkButton(
            master=self,
            text=">",
            width=25,
            height=25,
            command=lambda: _update_thresh_c_via_button(c_step),
        )

        self.thresh_c_name.place(x=1275, y=75)
        self.thresh_c.place(x=1385, y=80)
        self.thresh_c_label.place(x=1585, y=75)
        self.thresh_c_down.place(x=1620, y=75)
        self.thresh_c_up.place(x=1650, y=75)
        # ========== END C parameter ==========

        # ========== BEGIN Area parameter ==========
        min_area: int = 5
        max_area: int = 50
        area_step: int = 1
        default_area: int = 10

        self.min_area_var = ctk.IntVar(value=default_area)
        self.last_min_area_var = ctk.IntVar(value=default_area)

        self.min_area_name = ctk.CTkLabel(
            master=self,
            text="Min area (hair)",
            height=25,
            width=110,
        )

        def _update_min_area(position: float):
            position = int(position)

            if self.last_min_area_var.get() != position:
                self.last_min_area_var.set(position)
                self.min_area_label.configure(text=str(position))
                self._update_state()

        self.min_area = ctk.CTkSlider(
            master=self,
            from_=min_area,
            to=max_area,
            number_of_steps=(max_area - min_area) // area_step,
            variable=self.min_area_var,
            command=_update_min_area,
        )

        self.min_area_label = ctk.CTkLabel(
            master=self,
            text=str(self.min_area_var.get()),
        )

        def _update_min_area_via_button(change: int):
            new_value = self.min_area_var.get() + change

            if self.min_area.cget('from_') <= new_value <= self.min_area.cget('to'):
                self.min_area_var.set(new_value)
                self.min_area_label.configure(text=new_value)
                self._update_state()

        self.min_area_down = ctk.CTkButton(
            master=self,
            text="<",
            width=25,
            height=25,
            command=lambda: _update_min_area_via_button(-area_step),
        )

        self.min_area_up = ctk.CTkButton(
            master=self,
            text=">",
            width=25,
            height=25,
            command=lambda: _update_min_area_via_button(area_step),
        )

        self.min_area_name.place(x=1275, y=105)
        self.min_area.place(x=1385, y=110)
        self.min_area_label.place(x=1585, y=105)
        self.min_area_down.place(x=1620, y=105)
        self.min_area_up.place(x=1650, y=105)
        # ========== END Area parameter ==========

        # ========== BEGIN Dimension parameter ==========
        min_dim: int = 5
        max_dim: int = 100
        dim_step: int = 1
        default_dim: int = 10

        self.min_dim_var = ctk.IntVar(value=default_dim)
        self.last_min_dim_var = ctk.IntVar(value=default_dim)

        self.min_dim_name = ctk.CTkLabel(
            master=self,
            text="Min dim. (hair)",
            height=25,
            width=110,
        )

        def _update_min_dim(position: float):
            position = int(position)

            if self.last_min_dim_var.get() != position:
                self.last_min_dim_var.set(position)
                self.min_dim_label.configure(text=str(position))
                self._update_state()

        self.min_dim = ctk.CTkSlider(
            master=self,
            from_=min_dim,
            to=max_dim,
            number_of_steps=(max_dim - min_dim) // dim_step,
            variable=self.min_dim_var,
            command=_update_min_dim,
        )

        self.min_dim_label = ctk.CTkLabel(
            master=self,
            text=str(self.min_dim_var.get()),
        )

        def _update_min_dim_via_button(change: int):
            new_value = self.min_dim_var.get() + change

            if self.min_dim.cget('from_') <= new_value <= self.min_dim.cget('to'):
                self.min_dim_var.set(new_value)
                self.min_dim_label.configure(text=new_value)
                self._update_state()

        self.min_dim_down = ctk.CTkButton(
            master=self,
            text="<",
            width=25,
            height=25,
            command=lambda: _update_min_dim_via_button(-dim_step),
        )

        self.min_dim_up = ctk.CTkButton(
            master=self,
            text=">",
            width=25,
            height=25,
            command=lambda: _update_min_dim_via_button(dim_step),
        )

        self.min_dim_name.place(x=1275, y=135)
        self.min_dim.place(x=1385, y=140)
        self.min_dim_label.place(x=1585, y=135)
        self.min_dim_down.place(x=1620, y=135)
        self.min_dim_up.place(x=1650, y=135)
        # ========== END Area parameter ==========

        # ========== BEGIN Input directory ==========
        self.input_dir_var = ctk.StringVar(value="")

        self.input_dir = ctk.CTkEntry(
            master=self,
            placeholder_text="Directory with images",
            width=250,
            textvariable=self.input_dir_var,
        )

        def _choose_input_dir():
            cwd = os.getcwd()
            chosen = filedialog.askdirectory(parent=self, initialdir=os.getcwd(), mustexist=True)

            self.input_dir_var.set(
                chosen[len(cwd) + 1:] if chosen.startswith(cwd) else chosen,
            )

        self.input_dir_button = ctk.CTkButton(
            master=self,
            text="Choose",
            width=50,
            command=_choose_input_dir
        )

        self.input_dir.place(x=1275, y=185)
        self.input_dir_button.place(x=1540, y=185)
        # ========== END Input directory ==========

        # ========== BEGIN Output directory ==========
        self.output_dir_var = ctk.StringVar(value="")

        self.output_dir = ctk.CTkEntry(
            master=self,
            placeholder_text="Directory for masks to save",
            width=250,
            textvariable=self.output_dir_var,
        )

        def _choose_output_dir():
            cwd = os.getcwd()
            chosen = filedialog.askdirectory(parent=self, initialdir=os.getcwd(), mustexist=True)

            self.output_dir_var.set(
                chosen[len(cwd) + 1:] if chosen.startswith(cwd) else chosen,
            )

        self.output_dir_button = ctk.CTkButton(
            master=self,
            text="Choose",
            width=50,
            command=_choose_output_dir
        )

        self.output_dir.place(x=1275, y=220)
        self.output_dir_button.place(x=1540, y=220)
        # ========== END Output directory ==========

        # ========== BEGIN Save button ==========
        def _save_result():
            filepath = ""

            if not self.output_dir_var.get().startswith(os.sep):
                filepath = os.getcwd()

            filepath += os.sep + self.output_dir_var.get()

            rel_filepath = (self.orig_filepath[len(os.getcwd()) + 1:] if self.orig_filepath.startswith(os.getcwd()) else self.orig_filepath)
            rel_filepath = rel_filepath[len(self.input_dir_var.get()) + 1:] if rel_filepath.startswith(self.input_dir_var.get()) else rel_filepath

            shaved_filepath = filepath + os.sep + rel_filepath

            if not os.path.exists(os.path.dirname(shaved_filepath)):
                os.makedirs(os.path.dirname(shaved_filepath))

            ctkimage: ctk.CTkImage = self.res_label.cget("image")
            result: Image = ctkimage.cget("light_image")
            result.save(os.path.join(shaved_filepath), subsampling=0, quality=100)

            settings_filepath = filepath + "-settings"

            if not os.path.exists(settings_filepath):
                os.makedirs(settings_filepath)

            with open(settings_filepath + os.sep + os.path.basename(rel_filepath)[0:-4] + ".yaml", "w") as fp:
                fp.writelines([
                    "block_size: %.0f\n" % self.thresh_block_size_var.get(),
                    "c: %.0f\n" % self.thresh_c_var.get(),
                    "min_area: %.0f\n" % self.min_area_var.get(),
                    "min_dim: %.0f\n" % self.min_dim_var.get(),
                ])

                print("Saved %s (block size %d c %d min area %d min dim %d)" % (
                    os.path.basename(rel_filepath)[0:-4],
                    self.thresh_block_size_var.get(),
                    self.thresh_c_var.get(),
                    self.min_area_var.get(),
                    self.min_dim_var.get(),
                ))

        self.save_button = ctk.CTkButton(
            master=self,
            text="Save processed",
            command=_save_result
        )

        self.save_button.place(x=1275, y=260)
        # ========== END Save button ==========

        # ========== BEGIN No Hair ==========
        def _no_hair():
            input_dir = self.input_dir_var.get()
            output_dir = self.output_dir_var.get()

            rel_filepath = (self.orig_filepath[len(os.getcwd()) + 1:] if self.orig_filepath.startswith(os.getcwd()) else self.orig_filepath)
            rel_filepath = rel_filepath[len(input_dir) + 1:] if rel_filepath.startswith(input_dir) else rel_filepath

            source_path = os.path.join(os.getcwd(), input_dir, rel_filepath)
            target_path = os.path.join(output_dir, rel_filepath)

            if not os.path.islink(target_path):
                os.symlink(source_path, target_path)

                with open(output_dir + "-settings" + os.sep + os.path.basename(rel_filepath)[0:-4] + ".yaml", "w"):
                    pass

        self.no_hair_button = ctk.CTkButton(
            master=self,
            text="No hair",
            command=_no_hair,
        )

        self.no_hair_button.place(x=1425, y=260)
        # ========== END No Hair ==========

        # ========== BEGIN Load Random Image ==========
        def _load_random_image():
            files = glob.glob(os.path.join(self.input_dir_var.get(), '*', '*', '*.jpg'))
            files_len = len(files)

            while True:
                picked = files[random.randint(0, files_len - 1)]
                dirname = os.path.join(self.output_dir_var.get(), os.path.dirname(picked)[len(self.input_dir_var.get()) + 1:])
                basename = os.path.basename(picked)

                if not os.path.exists(os.path.join(dirname, basename)):
                    break

            self.thresh_block_size_var.set(default_block_size)
            self.thresh_c_var.set(default_c)
            self.min_area_var.set(default_area)
            self.min_dim_var.set(default_dim)

            self._update_window_state_on_dnd(SimpleNamespace(**{"data": picked}))

        self.load_random_image_button = ctk.CTkButton(
            master=self,
            text="Load random image",
            command=_load_random_image
        )

        self.load_random_image_button.place(x=1275, y=310)
        # ========== END Load Random Image ==========

        # ========== BEGIN Sync Data With Upload Frame ==========
        def _sync_data():
            if self._sync_method is not None:
                img: ctk.CTkImage = self.res_label.cget("image")
                img: Image.Image = img.cget("light_image")
                img.save("shaved.png")
                self._sync_method(img, self.orig_filepath)

            self.destroy()

        self.syn_data_button = ctk.CTkButton(
            master=self,
            text="Sync Data",
            command=_sync_data
        )

        self.syn_data_button.place(x=1425, y=310)
        # ========== END Sync Data With Upload Frame ==========

        self.orig_label.place(x=25, y=25)
        self.res_label.place(x=650, y=25)
        self.thresh_label.place(x=25, y=500)
        self.cc_label.place(x=650, y=500)
        self.cc_dil_label.place(x=1275, y=500)

        self.orig_label.drop_target_register(DND_FILES)
        self.orig_label.dnd_bind('<<Drop>>', lambda e: self._update_window_state_on_dnd(e))

        if self.orig_filepath != "":
            self._update_state()

    def set_filepath(self, filepath: str) -> None:
        self._load_image(filepath)
        self._update_state()

    def set_sync_method(self, method: Callable[[Image.Image, str], None]):
        self._sync_method = method

    def _update_window_state_on_dnd(self, e):
        filepath = str(e.data)
        self._load_image(filepath)
        self._update_state()

    def _load_image(self, filepath: str):
        img = Image.open(filepath)

        self.orig_filepath = filepath
        self.orig_img = img

        self.orig_label.configure(
            image=ctk.CTkImage(
                light_image=img,
                dark_image=img,
                size=(600, 450),
            ),
        )

    def _update_thresh_block_size(self, position: float):
        if self.last_thresh_block_size_var.get() != position:
            self.last_thresh_block_size_var.set(int(position))
            self.thresh_block_size_label.configure(text="%.0f" % position)
            self._update_state()

    def _update_thresh_block_size_via_button(self, change: int) -> None:
        new_value = self.thresh_block_size_var.get() + change

        if self.thresh_block_size.cget('from_') <= new_value <= self.thresh_block_size.cget('to'):
            self.thresh_block_size_var.set(new_value)
            self.thresh_block_size_label.configure(text=new_value)
            self._update_state()

    def _update_state(self):
        gray_img = cv2.cvtColor(np.array(self.orig_img), cv2.COLOR_RGB2GRAY)
        thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self.thresh_block_size_var.get(), self.thresh_c_var.get())

        self.thresh_label.configure(
            image=ctk.CTkImage(
                light_image=Image.fromarray(thresh_img, "L"),
                dark_image=Image.fromarray(thresh_img, "L"),
                size=(600, 450),
            ),
        )

        (found_components, component_ids, values, _) = cv2.connectedComponentsWithStats(thresh_img, None, cv2.CV_8U)
        mask = np.zeros(gray_img.shape, dtype=np.uint8)

        # Iterate over found connected components
        for i in range(1, found_components):
            # Dimension stats of the connected component
            area = values[i, cv2.CC_STAT_AREA]
            width = values[i, cv2.CC_STAT_WIDTH]
            height = values[i, cv2.CC_STAT_HEIGHT]

            # Add to the final mask only those connected components those dimensions and area exceed minimum values
            if (area > self.min_area_var.get()) and (width > self.min_dim_var.get() or height > self.min_dim_var.get()):
                component_mask = (component_ids == i).astype(np.uint8) * 255
                mask = cv2.bitwise_or(mask, component_mask)

        self.cc_label.configure(
            image=ctk.CTkImage(
                light_image=Image.fromarray(mask, "L"),
                dark_image=Image.fromarray(mask, "L"),
                size=(600, 450),
            ),
        )

        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [3, 3]))

        self.cc_dil_label.configure(
            image=ctk.CTkImage(
                light_image=Image.fromarray(mask, "L"),
                dark_image=Image.fromarray(mask, "L"),
                size=(600, 450),
            ),
        )

        res_img = cv2.inpaint(np.array(self.orig_img), mask, 7, cv2.INPAINT_TELEA)

        self.res_label.configure(
            image=ctk.CTkImage(
                light_image=Image.fromarray(res_img, "RGB"),
                dark_image=Image.fromarray(res_img, "RGB"),
                size=(600, 450),
            ),
        )
