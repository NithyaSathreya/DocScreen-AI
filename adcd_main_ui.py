import os
import json
import sys
import cv2
import threading
import numpy as np
import tkinter as tk

from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from modules.face import detect_faces
from modules.mrz import extract_mrz
from modules.pan import extract_pan
from modules.aadhaar import extract_aadhaar

try:
    from modules.ocr import extract_text
except Exception:
    extract_text = None

try:
    from modules.adcd import detect_tampering
except Exception:
    detect_tampering = None

ROOT = os.path.dirname(os.path.abspath(__file__))


def run_ocr(image_path, lang):
    if extract_text is None:
        return {"status": "unavailable", "text": None}
    try:
        return {"status": "success", "text": extract_text(image_path, lang)}
    except Exception as e:
        return {"status": "failed", "text": None, "error": str(e)}


def run_face(image_path):
    try:
        faces = detect_faces(image_path)
        return {"status": "success", "count": len(faces), "faces": faces}
    except Exception as e:
        return {"status": "failed", "count": 0, "faces": [], "error": str(e)}


def run_mrz(image_path):
    try:
        data = extract_mrz(image_path)
        return {"status": "success" if data else "not_detected", "data": data}
    except Exception as e:
        return {"status": "failed", "data": None, "error": str(e)}


def run_pan(image_path):
    try:
        data = extract_pan(image_path)
        return {"status": "success" if data else "not_detected", "data": data}
    except Exception as e:
        return {"status": "failed", "data": None, "error": str(e)}


def run_aadhaar(image_path):
    try:
        data = extract_aadhaar(image_path)
        return {"status": "success" if data else "not_detected", "data": data}
    except Exception as e:
        return {"status": "failed", "data": None, "error": str(e)}


def run_adcd(image_path, output_dir, gpu):
    if detect_tampering is None:
        return {"status": "unavailable", "error": "modules.adcd could not be imported."}
    try:
        return {"status": "success", "result": detect_tampering(image_path, output_dir, gpu)}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


class DocScreenGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DocScreen - AI Document Screening")
        self.root.geometry("1100x700")
        self.root.resizable(False, False)
        self.image_path = None
        self.output_dir = os.path.join(ROOT, "output")
        self.lang = tk.StringVar(value="en")
        self.gpu = tk.StringVar(value="0")
        self.document_type = tk.StringVar(value="Passport")
        self.status = tk.StringVar(value="Ready")
        self.build_ui()

    def build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Document").pack(side="left")
        ttk.Combobox(top, width=12, textvariable=self.document_type, values=["Passport", "PAN", "Aadhaar"], state="readonly").pack(side="left", padx=5)

        ttk.Label(top, text="OCR Language").pack(side="left", padx=(20, 0))
        ttk.Combobox(top, width=8, textvariable=self.lang, values=["en", "fr", "de", "es"], state="readonly").pack(side="left", padx=5)

        ttk.Label(top, text="GPU").pack(side="left", padx=(20, 0))
        ttk.Combobox(top, width=5, textvariable=self.gpu, values=["0", "-1"], state="readonly").pack(side="left", padx=5)

        ttk.Button(top, text="Select Image", command=self.select_image).pack(side="right")

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)

        left = ttk.LabelFrame(body, text="Image Preview", padding=10)
        left.pack(side="left", fill="both", padx=10, pady=5)

        self.image_label = ttk.Label(left)
        self.image_label.pack()

        self.file_label = ttk.Label(left, text="No image selected", wraplength=350)
        self.file_label.pack(pady=10)

        right = ttk.LabelFrame(body, text="Screening Results", padding=10)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=5)

        self.result_box = tk.Text(right, width=55, height=28, font=("Consolas", 10))
        self.result_box.pack(fill="both")

        bottom = ttk.Frame(self.root, padding=10)
        bottom.pack(fill="x")

        self.run_btn = ttk.Button(bottom, text="RUN SCREENING", command=self.start_screening)
        self.run_btn.pack(fill="x", ipady=8)

        ttk.Label(self.root, textvariable=self.status, relief="sunken", anchor="w").pack(fill="x", side="bottom")

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if not path:
            return
        self.image_path = path
        self.file_label.config(text=path)
        self.show_image(path)
        self.result_box.delete("1.0", "end")
        self.status.set("Image selected")

    def show_image(self, path):
        img = Image.open(path)
        img.thumbnail((420, 420))
        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_img)

    def log(self, text):
        self.result_box.insert("end", text + "\n")
        self.result_box.see("end")
        self.root.update_idletasks()

    def start_screening(self):
        if not self.image_path:
            messagebox.showerror("Error", "Select an image first.")
            return
        self.result_box.delete("1.0", "end")
        self.run_btn.config(state="disabled")
        self.status.set("Screening...")
        threading.Thread(target=self.screen, daemon=True).start()

    def screen(self):
        os.makedirs(self.output_dir, exist_ok=True)
        document_type = self.document_type.get()
        report = {"input": self.image_path, "document_type": document_type}

        self.log("========== OCR ==========")
        ocr = run_ocr(self.image_path, self.lang.get())
        report["ocr"] = ocr
        self.log(f"Status : {ocr['status']}")

        if document_type == "Passport":
            self.log("\n========== PASSPORT / MRZ ==========")
            mrz = run_mrz(self.image_path)
            report["mrz"] = mrz
            self.log(f"Status : {mrz['status']}")
            if mrz.get("data"):
                d = mrz["data"]
                self.log(f"Passport : {d.get('passport_number')}")
                self.log(f"Name     : {d.get('surname')}, {d.get('given_names')}")
                self.log(f"Nationality : {d.get('nationality')}")
                self.log(f"DOB      : {d.get('date_of_birth')}")
                self.log(f"Expiry   : {d.get('expiration_date')}")
                self.log(f"Sex      : {d.get('sex')}")
                self.log(f"Score    : {d.get('valid_score')}")

        elif document_type == "PAN":
            self.log("\n========== PAN ==========")
            pan = run_pan(self.image_path)
            report["pan"] = pan
            self.log(f"Status : {pan['status']}")
            if pan.get("data"):
                d = pan["data"]
                self.log(f"PAN    : {d.get('pan_number')}")
                self.log(f"Valid  : {d.get('valid_number')}")

        elif document_type == "Aadhaar":
            self.log("\n========== AADHAAR ==========")
            aadhaar = run_aadhaar(self.image_path)
            report["aadhaar"] = aadhaar
            self.log(f"Status : {aadhaar['status']}")
            if aadhaar.get("data"):
                d = aadhaar["data"]
                self.log(f"Aadhaar : {d.get('aadhaar_number')}")
                self.log(f"Valid   : {d.get('valid_number')}")

        self.log("\n========== FACE ==========")
        face = run_face(self.image_path)
        report["face"] = face
        self.log(f"Status : {face['status']}")
        self.log(f"Faces  : {face['count']}")

        self.log("\n========== ADCD-NET ==========")
        adcd = run_adcd(self.image_path, self.output_dir, self.gpu.get())
        report["adcd"] = adcd
        self.log(f"Status : {adcd['status']}")

        heatmap = None
        if adcd.get("result"):
            a = adcd["result"]
            heatmap = a["heatmap"]
            self.log(f"Tampering Score : {a['score']:.4f}")
            self.log(f"Maximum Score  : {a['max_score']:.4f}")
            self.log(f"DCT Alignment   : {a['alignment_score']:.4f}")
            self.log(f"Result          : {a['result']}")
            self.log(f"Heatmap         : {a['heatmap']}")
            self.log(f"Mask            : {a['mask']}")

        report_path = os.path.join(self.output_dir, os.path.splitext(os.path.basename(self.image_path))[0] + "_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        self.log("\n===================================")
        self.log("SCREENING COMPLETE")
        self.log(report_path)

        self.status.set("Completed")
        self.run_btn.config(state="normal")

        if heatmap:
            self.show_image(heatmap)


if __name__ == "__main__":
    root = tk.Tk()
    DocScreenGUI(root)
    root.mainloop()
